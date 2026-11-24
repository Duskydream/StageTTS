import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Upload } from "lucide-react";
import type React from "react";
import { memo, useRef } from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { type StageFlowNode, useWorkflowStore } from "../../store/useWorkflowStore";
import type { StemVolumes, SubtitleBurnMode } from "../../types/script";
import type { NodeRunStatus } from "../../types/workflow";

const STATUS_DOT: Record<NodeRunStatus, string> = {
  idle: "bg-[#666666]",
  queued: "bg-status-warning",
  running: "bg-status-ready animate-pulse",
  done: "bg-status-ready",
  error: "bg-status-error",
};

const NEUTRAL_PORT_CLASS = "!h-2 !w-2 !border !bg-[#7A7A7A] !border-[#181818]";

function formatDurationShort(sec: number | undefined): string {
  if (!sec || !Number.isFinite(sec)) return "00:00";
  const mins = Math.floor(sec / 60);
  const secs = Math.floor(sec % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

const NodeInlineOptions: React.FC<{ nodeId: string }> = ({ nodeId }) => {
  const script = useProjectStore((s) => s.script);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const importMediaFile = useProjectStore((s) => s.importMediaFile);
  const updateTranslationSettings = useProjectStore((s) => s.updateTranslationSettings);
  const updateBgmVolume = useProjectStore((s) => s.updateBgmVolume);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);

  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const updatePipelineConfig = useWorkflowStore((s) => s.updatePipelineConfig);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const disabled = isPipelineRunning || isLoading || isSynthesizing;
  const chip = "rounded-[3px] bg-white/[0.04] px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-secondary";
  const selectCls =
    "h-6 min-w-0 flex-1 rounded-[3px] border border-white/[0.08] bg-[#121212] px-1.5 text-[11px] text-primary outline-none disabled:opacity-50";

  if (nodeId === "media_input") {
    return (
      <div className="nodrag nopan flex items-center justify-between gap-2 border-t border-white/[0.06] px-2.5 py-1.5 text-[11px]">
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void importMediaFile(file);
            e.target.value = "";
          }}
        />
        <span className="min-w-0 truncate font-mono tabular-nums text-secondary" title={script?.project_id}>
          {script ? `${script.project_id} · ${formatDurationShort(script.media.duration)}` : "未导入"}
        </span>
        <button
          type="button"
          disabled={disabled}
          title="导入 / 更换音视频"
          onClick={() => fileInputRef.current?.click()}
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[3px] border border-white/[0.08] bg-[#222222] text-secondary transition-colors hover:bg-[#2E2E2E] hover:text-primary disabled:opacity-40"
        >
          <Upload className="h-3 w-3" strokeWidth={1.6} />
        </button>
      </div>
    );
  }

  if (nodeId === "banquet") {
    const stems = script?.media.cinematic_stems;
    const items = [
      { key: "dialogue", label: "对", ready: Boolean(stems?.dialogue || script?.media.vocals_wav) },
      { key: "singing", label: "歌", ready: Boolean(stems?.singing) },
      { key: "music", label: "乐", ready: Boolean(stems?.music || script?.media.bgm_wav) },
      { key: "effects", label: "效", ready: Boolean(stems?.effects) },
    ];
    return (
      <div className="nodrag nopan flex items-center justify-between border-t border-white/[0.06] px-2.5 py-1.5 text-[10px]">
        <span className="text-muted">Banquet</span>
        <div className="flex items-center gap-2">
          {items.map((item) => (
            <span key={item.key} className="flex items-center gap-1 text-secondary">
              <span
                className={`h-1.5 w-1.5 rounded-full ${item.ready ? "bg-status-ready" : "bg-[#4A4A4A]"}`}
              />
              <span>{item.label}</span>
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (nodeId === "whisperx") {
    return (
      <div className="nodrag nopan flex items-center gap-1 border-t border-white/[0.06] px-2.5 py-1.5">
        <span className={chip}>MOSS</span>
        <span className={chip}>BF16</span>
        <span className={chip}>≤90m</span>
      </div>
    );
  }

  if (nodeId === "llm_tagger") {
    return (
      <div className="nodrag nopan space-y-1.5 border-t border-white/[0.06] px-2.5 py-2 text-[11px]">
        <div className="flex items-center gap-1.5">
          <span className="w-7 shrink-0 text-[10px] text-muted">目标</span>
          <select
            aria-label="管线目标语言"
            value={pipelineConfig.targetLang}
            disabled={isPipelineRunning}
            onChange={(e) =>
              updatePipelineConfig({ targetLang: e.target.value as "zh" | "en" | "ja" })
            }
            className={selectCls}
          >
            <option value="zh">中文</option>
            <option value="en">英文</option>
            <option value="ja">日语</option>
          </select>
        </div>
        {script && (
          <div className="flex items-center gap-1.5">
            <span className="w-7 shrink-0 text-[10px] text-muted">原声</span>
            <select
              aria-label="原声语言"
              value={script.source_lang ?? ""}
              disabled={disabled}
              onChange={(e) =>
                updateTranslationSettings(e.target.value || null, script.glossary ?? {})
              }
              className={selectCls}
            >
              <option value="">待定</option>
              <option value="ja">日语</option>
              <option value="en">英语</option>
              <option value="zh">普通话</option>
              <option value="ko">韩语</option>
              <option value="yue">粤语</option>
            </select>
          </div>
        )}
        <div className="flex items-center justify-between gap-2 pt-0.5 text-[10px] text-secondary">
          <label
            className="flex cursor-pointer items-center gap-1"
            title="翻译前通读全片提取人物性格与术语，翻译后按配音气口做口语润色"
          >
            <input
              type="checkbox"
              aria-label="口语润色"
              disabled={isPipelineRunning}
              checked={pipelineConfig.naturalPolish}
              onChange={(e) => updatePipelineConfig({ naturalPolish: e.target.checked })}
              className="h-3 w-3"
            />
            <span>口语润色</span>
          </label>
          <label
            className="flex cursor-pointer items-center gap-1"
            title="翻译前对人名、作品专名与网络典故执行免费联网检索校对译名"
          >
            <input
              type="checkbox"
              aria-label="联网校对"
              disabled={isPipelineRunning}
              checked={pipelineConfig.webSearch}
              onChange={(e) => updatePipelineConfig({ webSearch: e.target.checked })}
              className="h-3 w-3"
            />
            <span>联网校对</span>
          </label>
        </div>
      </div>
    );
  }

  if (nodeId === "script_hub") {
    const segCount = script?.segments.length ?? 0;
    const charCount = script ? Object.keys(script.characters).length : 0;
    const readyCount = script
      ? script.segments.filter((seg) => seg.keep_original || seg.status === "ready").length
      : 0;
    return (
      <div className="nodrag nopan flex items-center gap-1 border-t border-white/[0.06] px-2.5 py-1.5">
        <span className={chip}>{segCount} 句</span>
        <span className={chip}>{charCount} 角</span>
        <span className={chip}>{readyCount} 绪</span>
      </div>
    );
  }

  if (nodeId === "agent_tts") {
    const pending = script
      ? script.segments.filter((seg) => !seg.keep_original && seg.status !== "ready").length
      : 0;
    return (
      <div className="nodrag nopan flex items-center justify-between border-t border-white/[0.06] px-2.5 py-1.5 text-[10px]">
        <span className={pending > 0 ? "font-mono text-status-warning" : "font-mono text-status-ready"}>
          {script ? (pending > 0 ? `待配 ${pending}` : "已就绪") : "GPT-SoVITS"}
        </span>
        <span className={chip}>v2Pro</span>
      </div>
    );
  }

  if (nodeId === "ffmpeg_mixer") {
    const bgm = script?.media.bgm_volume ?? 0.75;
    const stems = script?.media.cinematic_stems;
    const stemItems: Array<{ key: keyof StemVolumes; label: string }> = [
      { key: "singing", label: "歌" },
      { key: "music", label: "乐" },
      { key: "effects", label: "效" },
    ];
    return (
      <div className="nodrag nopan space-y-1.5 border-t border-white/[0.06] px-2.5 py-2 text-[11px]">
        <div className="flex items-center gap-1.5">
          <span className="w-7 shrink-0 text-[10px] text-muted">伴奏</span>
          <input
            aria-label="伴奏主音量"
            type="range"
            min={0}
            max={2}
            step={0.05}
            disabled={!script || disabled}
            value={bgm}
            onChange={(e) => updateBgmVolume(Number(e.target.value))}
            className="min-w-0 flex-1"
          />
          <span className="w-8 shrink-0 text-right font-mono tabular-nums text-[10px] text-secondary">
            {Math.round(bgm * 100)}%
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="w-7 shrink-0 text-[10px] text-muted">字幕</span>
          <select
            aria-label="烧录 ASS 字幕"
            value={pipelineConfig.burnSubtitles}
            disabled={isPipelineRunning}
            onChange={(e) =>
              updatePipelineConfig({ burnSubtitles: e.target.value as SubtitleBurnMode })
            }
            className={selectCls}
          >
            <option value="none">无</option>
            <option value="source">原文（识别）</option>
            <option value="target">译文</option>
            <option value="bilingual">双语</option>
          </select>
        </div>

        <div className="flex items-center justify-between gap-2 pt-0.5 text-[10px] text-secondary">
          <label className="flex cursor-pointer items-center gap-1">
            <input
              type="checkbox"
              disabled={!script || disabled}
              checked={script?.media.auto_ducking ?? true}
              onChange={(e) => updateAutoDucking(e.target.checked)}
              className="h-3 w-3"
            />
            <span>闪避</span>
          </label>
          <label className="flex cursor-pointer items-center gap-1" title="导出时烧录“AI 合成语音”水印（重编码）">
            <input
              type="checkbox"
              disabled={isPipelineRunning}
              checked={pipelineConfig.burnWatermark}
              onChange={(e) => updatePipelineConfig({ burnWatermark: e.target.checked })}
              className="h-3 w-3"
            />
            <span>水印</span>
          </label>
          {stems && (
            <details className="text-[10px] text-muted">
              <summary className="cursor-pointer select-none hover:text-primary">四轨</summary>
              <div className="mt-1.5 space-y-1">
                {stemItems.map((item) => {
                  const val = script?.media.stem_volumes?.[item.key] ?? 1;
                  return (
                    <div key={item.key} className="flex items-center gap-1">
                      <span className="w-4 shrink-0 text-secondary">{item.label}</span>
                      <input
                        aria-label={`${item.label}轨`}
                        type="range"
                        min={0}
                        max={2}
                        step={0.05}
                        disabled={disabled}
                        value={val}
                        onChange={(e) => updateStemVolume(item.key, Number(e.target.value))}
                        className="w-24"
                      />
                      <span className="w-7 text-right font-mono tabular-nums">{Math.round(val * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </details>
          )}
        </div>
      </div>
    );
  }

  return null;
};

/**
 * ComfyUI 极简中性节点卡片（无分类色标记，仅保留状态指示与连线配色）。
 */
export const StageNode: React.FC<NodeProps<StageFlowNode>> = memo(({ id, data, selected }) => {
  const rowCount = Math.max(data.inputs.length, data.outputs.length, 1);
  const borderColor =
    data.status === "running"
      ? "#4FA882"
      : data.status === "error"
        ? "#C95252"
        : selected
          ? "rgba(255, 255, 255, 0.45)"
          : "rgba(255, 255, 255, 0.1)";

  return (
    <div
      style={{
        borderColor,
        boxShadow: selected ? "0 0 0 1px rgba(255, 255, 255, 0.28)" : undefined,
      }}
      className="w-[244px] overflow-hidden rounded-[6px] border bg-[#181818] transition-colors"
    >
      {/* 顶栏（#1E1E1E 中性暗灰 + 状态点 + VRAM 数值） */}
      <div className="flex items-center justify-between border-b border-white/[0.06] bg-[#1E1E1E] px-2.5 py-1.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[data.status]}`} />
          <span className="truncate text-[12px] font-medium text-primary">{data.title}</span>
        </div>
        <span className="ml-2 shrink-0 rounded-[3px] bg-white/[0.05] px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-secondary">
          {data.vramGb === null ? "—" : data.vramGb > 0 ? `${data.vramGb.toFixed(1)}G` : "CPU"}
        </span>
      </div>

      {/* 左右端口区 */}
      <div className="space-y-1.5 py-2 text-[11px]">
        {Array.from({ length: rowCount }).map((_, idx) => {
          const inputPort = data.inputs[idx];
          const outputPort = data.outputs[idx];
          return (
            <div
              key={`${inputPort?.id ?? "in"}-${outputPort?.id ?? "out"}-${idx}`}
              className="relative flex min-h-[18px] items-center justify-between px-3"
            >
              {inputPort ? (
                <div className="flex items-center gap-1.5">
                  <Handle
                    id={inputPort.id}
                    type="target"
                    position={Position.Left}
                    className={NEUTRAL_PORT_CLASS}
                  />
                  <span className="text-secondary">{inputPort.label}</span>
                </div>
              ) : (
                <span />
              )}

              {outputPort ? (
                <div className="flex items-center gap-1.5 text-right">
                  <span className="text-primary">{outputPort.label}</span>
                  <Handle
                    id={outputPort.id}
                    type="source"
                    position={Position.Right}
                    className={NEUTRAL_PORT_CLASS}
                  />
                </div>
              ) : (
                <span />
              )}
            </div>
          );
        })}
      </div>

      <NodeInlineOptions nodeId={id} />
    </div>
  );
});

StageNode.displayName = "StageNode";
