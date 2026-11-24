import { CheckCircle2, Download, ExternalLink, FolderOpen, X } from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";
import {
  ApiError,
  type ExportDirsResponse,
  fetchExportDirs,
  getSubtitlesUrl,
  renderVideo,
} from "../../services/api";
import { getCacheBustedUrl, useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type {
  RenderResponse,
  StemVolumes,
  SubtitleBurnMode,
  SubtitleLayout,
} from "../../types/script";

interface ExportModalProps {
  open: boolean;
  onClose: () => void;
  onExportSuccess?: (message: string) => void;
}

interface SaveFilePickerWindow {
  showSaveFilePicker?: (options?: {
    suggestedName?: string;
    types?: Array<{ description?: string; accept: Record<string, string[]> }>;
  }) => Promise<{
    createWritable: () => Promise<
      {
        write: (data: Blob) => Promise<void>;
        close: () => Promise<void>;
        abort?: () => Promise<void>;
      } & WritableStream<Uint8Array>
    >;
  }>;
}

type SidecarOption = "none" | SubtitleLayout;

function defaultFilename(projectId: string, isAudioOnly: boolean): string {
  return `${projectId}_dubbed.${isAudioOnly ? "wav" : "mp4"}`;
}

export const ExportModal: React.FC<ExportModalProps> = ({ open, onClose, onExportSuccess }) => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isDirty = useProjectStore((s) => s.isDirty);
  const setIsPlaying = useProjectStore((s) => s.setIsPlaying);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);

  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const updatePipelineConfig = useWorkflowStore((s) => s.updatePipelineConfig);
  const appendLog = useWorkflowStore((s) => s.appendLog);

  const isAudioOnly = script?.mode === "virtual_stage" && !script?.media.source_video;
  const projectId = script?.project_id ?? "";

  const [filename, setFilename] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [sidecar, setSidecar] = useState<SidecarOption>("none");
  const [dirOptions, setDirOptions] = useState<ExportDirsResponse | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<RenderResponse | null>(null);
  const [exportStamp, setExportStamp] = useState<number>(0);

  useEffect(() => {
    if (!open || !projectId) {
      return;
    }
    setIsPlaying(false);
    setErrorMsg(null);
    setResult(null);
    setFilename(defaultFilename(projectId, Boolean(isAudioOnly)));
  }, [open, projectId, isAudioOnly, setIsPlaying]);

  useEffect(() => {
    if (!open || !projectId || backendStatus !== "online") {
      return;
    }
    void fetchExportDirs()
      .then((data) => setDirOptions(data))
      .catch(() => setDirOptions(null));
  }, [open, projectId, backendStatus]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isExporting) {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, isExporting, onClose]);

  if (!open || !script) {
    return null;
  }

  const stems = script.media.cinematic_stems;
  const stemVolumes = script.media.stem_volumes ?? { singing: 1, music: 1, effects: 1 };
  const autoDucking = script.media.auto_ducking ?? true;
  const needsReencode = pipelineConfig.burnSubtitles !== "none" || pipelineConfig.burnWatermark;

  const handleSaveAs = async (url: string, suggestedName: string) => {
    const picker = (window as unknown as SaveFilePickerWindow).showSaveFilePicker;
    if (picker) {
      let writable:
        | ({
            write: (data: Blob) => Promise<void>;
            close: () => Promise<void>;
            abort?: () => Promise<void>;
          } & WritableStream<Uint8Array>)
        | null = null;
      try {
        const isWav = suggestedName.endsWith(".wav");
        const handle = await picker({
          suggestedName,
          types: [
            {
              description: isWav ? "WAV 音频" : "MP4 视频",
              accept: isWav ? { "audio/wav": [".wav"] } : { "video/mp4": [".mp4"] },
            },
          ],
        });
        const resp = await fetch(url);
        if (!resp.ok) {
          throw new Error("读取导出文件失败");
        }
        writable = await handle.createWritable();
        if (resp.body) {
          await resp.body.pipeTo(writable);
        } else {
          const blob = await resp.blob();
          await writable.write(blob);
          await writable.close();
        }
        return;
      } catch (err) {
        if (writable?.abort) {
          await writable.abort().catch(() => {});
        }
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        setErrorMsg(err instanceof Error ? err.message : "另存为写入失败，请检查磁盘空间或权限");
        return;
      }
    }
    const link = document.createElement("a");
    link.href = url;
    link.download = suggestedName;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleConfirmExport = async () => {
    if (isExporting) {
      return;
    }
    if (backendStatus === "offline") {
      setErrorMsg("当前处于离线预览模式，请连接后端后再导出成片。");
      return;
    }
    setIsExporting(true);
    setErrorMsg(null);
    setResult(null);
    try {
      if (isDirty) {
        await saveProjectScript();
        const afterSave = useProjectStore.getState();
        if (afterSave.error || afterSave.isDirty) {
          setErrorMsg(afterSave.error ?? "请先保存当前修改后再导出");
          return;
        }
      }
      const res = await renderVideo(script.project_id, {
        burn_watermark: isAudioOnly ? false : pipelineConfig.burnWatermark,
        burn_subtitles: isAudioOnly ? "none" : pipelineConfig.burnSubtitles,
        export_ass_sidecar: sidecar === "none" ? null : sidecar,
        auto_ducking: autoDucking,
        output_filename: filename.trim() || null,
        output_dir: outputDir.trim() || null,
      });
      useProjectStore.setState((state) =>
        state.script?.project_id === res.project_id
          ? {
              script: {
                ...state.script,
                media: { ...state.script.media, output_video: res.output_video },
              },
            }
          : state,
      );
      setExportStamp(Date.now());
      setResult(res);
      appendLog(
        "env_backend",
        `导出成功: ${res.saved_path || res.output_video}${res.subtitle_saved_path ? ` + ${res.subtitle_saved_path}` : ""}`,
        "success",
      );
      onExportSuccess?.("已导出成片");
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "导出失败，请检查日志";
      setErrorMsg(msg);
      appendLog("env_backend", msg, "error");
    } finally {
      setIsExporting(false);
    }
  };

  const stemControls: Array<{ key: keyof StemVolumes; label: string }> = [
    { key: "singing", label: "歌声" },
    { key: "music", label: "音乐" },
    { key: "effects", label: "音效" },
  ];

  const burnOptions: Array<{ id: SubtitleBurnMode; label: string }> = [
    { id: "none", label: "不烧录" },
    { id: "source", label: "识别原文" },
    { id: "target", label: "单语译文" },
    { id: "bilingual", label: "双语对照" },
  ];

  const sidecarOptions: Array<{ id: SidecarOption; label: string }> = [
    { id: "none", label: "不导出" },
    { id: "source", label: "原文 .ass" },
    { id: "target", label: "单语 .ass" },
    { id: "bilingual", label: "双语 .ass" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 text-[12px]">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-modal-title"
        className="flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-[4px] border border-white/[0.1] bg-[#222222]"
      >
        {/* 标题栏 */}
        <div className="flex h-9 shrink-0 items-center justify-between border-b border-white/[0.06] bg-[#1C1C1C] px-3.5">
          <span id="export-modal-title" className="text-[13px] font-medium text-primary">
            导出成片与字幕设置
          </span>
          <button
            type="button"
            disabled={isExporting}
            onClick={onClose}
            aria-label="关闭导出窗口"
            className="flex h-6 w-6 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-40"
          >
            <X className="h-3.5 w-3.5" strokeWidth={1.5} />
          </button>
        </div>

        {/* 主体内容 */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {/* 1. 输出位置与文件名 */}
          <section className="space-y-2.5">
            <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
              1 · 输出文件与保存位置
            </div>
            <div className="space-y-1">
              <label htmlFor="export-filename" className="block text-[11px] text-secondary">
                输出文件名
              </label>
              <input
                id="export-filename"
                type="text"
                disabled={isExporting}
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                placeholder={defaultFilename(script.project_id, Boolean(isAudioOnly))}
                className="h-7 w-full rounded-[4px] border border-white/[0.08] bg-[#181818] px-2.5 font-mono text-[12px] text-primary focus:border-white/25 focus:outline-none disabled:opacity-50"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="export-dir" className="block text-[11px] text-secondary">
                额外保存到本机目录（留空则仅保存在项目工作区）
              </label>
              <div className="flex items-center gap-1.5">
                <FolderOpen className="h-3.5 w-3.5 shrink-0 text-secondary" strokeWidth={1.5} />
                <input
                  id="export-dir"
                  type="text"
                  disabled={isExporting}
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder={
                    dirOptions
                      ? `${dirOptions.workspace_dir}${dirOptions.workspace_dir.includes("\\") ? "\\" : "/"}${script.project_id}`
                      : "留空使用项目工作区，或点下方快捷目录 / 输入本机绝对路径"
                  }
                  className="h-7 min-w-0 flex-1 rounded-[4px] border border-white/[0.08] bg-[#181818] px-2.5 font-mono text-[11px] text-primary placeholder:text-muted focus:border-white/25 focus:outline-none disabled:opacity-50"
                />
              </div>
              <div className="flex flex-wrap gap-1.5 pt-0.5">
                <button
                  type="button"
                  disabled={isExporting}
                  onClick={() => setOutputDir("")}
                  className={`rounded-[3px] border px-2 py-0.5 text-[11px] transition-colors ${
                    outputDir.trim() === ""
                      ? "border-white/20 bg-[#333333] text-primary"
                      : "border-white/[0.08] bg-[#2A2A2A] text-secondary hover:text-primary"
                  }`}
                >
                  项目工作区
                </button>
                {dirOptions?.suggested_dirs.map((dir) => (
                  <button
                    key={dir.path}
                    type="button"
                    disabled={isExporting}
                    onClick={() => setOutputDir(dir.path)}
                    title={dir.path}
                    className={`rounded-[3px] border px-2 py-0.5 text-[11px] transition-colors ${
                      outputDir.trim() === dir.path
                        ? "border-white/20 bg-[#333333] text-primary"
                        : "border-white/[0.08] bg-[#2A2A2A] text-secondary hover:text-primary"
                    }`}
                  >
                    {dir.label}
                  </button>
                ))}
              </div>
            </div>
          </section>

          {/* 2. 字幕与画面编码 */}
          <section className="space-y-2.5 border-t border-dashed border-white/[0.08] pt-3">
            <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
              2 · ASS 字幕与画面编码
            </div>

            {!isAudioOnly && (
              <div className="space-y-1">
                <span className="block text-[11px] text-secondary">烧录 ASS 硬字幕到画面</span>
                <div className="grid h-7 grid-cols-4 gap-0.5 rounded-[4px] bg-[#181818] p-0.5">
                  {burnOptions.map((opt) => {
                    const active = pipelineConfig.burnSubtitles === opt.id;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        disabled={isExporting}
                        onClick={() => updatePipelineConfig({ burnSubtitles: opt.id })}
                        className={`rounded-[3px] text-[11px] transition-colors ${
                          active
                            ? "bg-[#333333] font-medium text-[#FFFFFF]"
                            : "text-secondary hover:text-primary"
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="space-y-1">
              <span className="block text-[11px] text-secondary">同步导出外挂 ASS 字幕文件</span>
              <div className="grid h-7 grid-cols-4 gap-0.5 rounded-[4px] bg-[#181818] p-0.5">
                {sidecarOptions.map((opt) => {
                  const active = sidecar === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      disabled={isExporting}
                      onClick={() => setSidecar(opt.id)}
                      className={`rounded-[3px] text-[11px] transition-colors ${
                        active
                          ? "bg-[#333333] font-medium text-[#FFFFFF]"
                          : "text-secondary hover:text-primary"
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {!isAudioOnly && (
              <label className="flex cursor-pointer items-center justify-between gap-2 pt-0.5 text-[11px] text-primary">
                <span>烧录“AI 合成语音”合规角标水印</span>
                <input
                  type="checkbox"
                  disabled={isExporting}
                  checked={pipelineConfig.burnWatermark}
                  onChange={(e) => updatePipelineConfig({ burnWatermark: e.target.checked })}
                  className="h-3.5 w-3.5"
                />
              </label>
            )}

            <p className="rounded-[3px] bg-[#181818] px-2.5 py-1.5 font-mono text-[11px] text-secondary">
              {isAudioOnly
                ? "当前为纯音频舞台模式，将混音导出立体声 WAV。"
                : needsReencode
                  ? "编码策略：单次重编码（自动启用硬件加速或 libx264），同时烧录已选滤镜。"
                  : "编码策略：视频流直拷贝 (-c:v copy)，秒级无损封装。"}
            </p>
          </section>

          {/* 3. 混音与四轨平衡 */}
          <section className="space-y-2 border-t border-dashed border-white/[0.08] pt-3">
            <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
              3 · 混音与闪避参数
            </div>

            <div className="flex items-center gap-2 text-[11px]">
              <span className="w-16 shrink-0 text-secondary">伴奏主音量</span>
              <input
                aria-label="导出伴奏主音量"
                type="range"
                disabled={isExporting}
                min={0}
                max={2}
                step={0.05}
                value={script.media.bgm_volume}
                onChange={(e) => updateBgmVolume(Number(e.target.value))}
                className="min-w-0 flex-1"
              />
              <span className="w-10 shrink-0 text-right font-mono tabular-nums text-primary">
                {Math.round(script.media.bgm_volume * 100)}%
              </span>
            </div>

            {stems &&
              stemControls.map((item) => {
                const value = stemVolumes[item.key] ?? 1;
                return (
                  <div key={item.key} className="flex items-center gap-2 text-[11px]">
                    <span className="w-16 shrink-0 text-secondary">{item.label}轨</span>
                    <input
                      aria-label={`导出${item.label}轨音量`}
                      type="range"
                      disabled={isExporting}
                      min={0}
                      max={2}
                      step={0.05}
                      value={value}
                      onChange={(e) => updateStemVolume(item.key, Number(e.target.value))}
                      className="min-w-0 flex-1"
                    />
                    <span className="w-10 shrink-0 text-right font-mono tabular-nums text-secondary">
                      {Math.round(value * 100)}%
                    </span>
                  </div>
                );
              })}

            <label className="flex cursor-pointer items-center justify-between gap-2 pt-0.5 text-[11px] text-primary">
              <span>人声出现时自动闪避背景音乐</span>
              <input
                type="checkbox"
                disabled={isExporting}
                checked={autoDucking}
                onChange={(e) => updateAutoDucking(e.target.checked)}
                className="h-3.5 w-3.5"
              />
            </label>
          </section>

          {/* 错误与导出结果 */}
          {errorMsg && (
            <div className="rounded-[4px] border border-[#C95252]/40 bg-[#C95252]/10 px-3 py-2 text-[11px] text-status-error">
              {errorMsg}
            </div>
          )}

          {result && (
            <div className="space-y-2 rounded-[4px] border border-[#4FA882]/40 bg-[#181818] p-3">
              <div className="flex items-center gap-1.5 text-[12px] font-medium text-status-ready">
                <CheckCircle2 className="h-4 w-4" strokeWidth={1.5} />
                <span>导出完成（{result.duration.toFixed(1)}s）</span>
              </div>
              <div className="space-y-1 font-mono text-[11px] text-secondary">
                <div className="break-all">
                  成片路径：<span className="text-primary">{result.saved_path}</span>
                </div>
                {result.subtitle_saved_path && (
                  <div className="break-all">
                    字幕路径：<span className="text-primary">{result.subtitle_saved_path}</span>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="button"
                  onClick={() =>
                    void handleSaveAs(
                      getCacheBustedUrl(result.output_video, exportStamp) ?? result.output_video,
                      result.output_video.split("/").pop() || "output.mp4",
                    )
                  }
                  className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 text-[11px] font-medium text-[#FFFFFF] hover:bg-[#467DB8]"
                >
                  <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
                  <span>另存为 / 下载成片</span>
                </button>
                <a
                  href={getCacheBustedUrl(result.output_video, exportStamp) ?? result.output_video}
                  target="_blank"
                  rel="noreferrer"
                  className="flex h-7 items-center gap-1 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 text-[11px] text-primary hover:bg-[#363636]"
                >
                  <ExternalLink className="h-3 w-3" strokeWidth={1.5} />
                  <span>打开预览</span>
                </a>
                {result.subtitle_url ? (
                  <a
                    href={
                      getCacheBustedUrl(result.subtitle_url, exportStamp) ?? result.subtitle_url
                    }
                    download={result.subtitle_url.split("/").pop() || "subtitles.ass"}
                    className="flex h-7 items-center rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 text-[11px] text-primary hover:bg-[#363636]"
                  >
                    下载同名字幕 (.ass)
                  </a>
                ) : (
                  <a
                    href={getSubtitlesUrl(script.project_id, "bilingual")}
                    download={`${script.project_id}_bilingual.ass`}
                    className="flex h-7 items-center rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 text-[11px] text-primary hover:bg-[#363636]"
                  >
                    下载双语 ASS
                  </a>
                )}
              </div>
            </div>
          )}
        </div>

        {/* 底部按钮 */}
        <div className="flex shrink-0 items-center justify-end gap-2 border-t border-white/[0.06] bg-[#1C1C1C] px-4 py-2.5">
          <button
            type="button"
            disabled={isExporting}
            onClick={onClose}
            className="h-7 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-3 text-[12px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
          >
            {result ? "完成" : "取消"}
          </button>
          <button
            type="button"
            disabled={isExporting}
            onClick={() => void handleConfirmExport()}
            className="flex h-7 items-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-3.5 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>{isExporting ? "导出处理中..." : result ? "重新导出" : "开始导出"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
