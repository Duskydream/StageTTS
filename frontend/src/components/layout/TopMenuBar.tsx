import { useQuery } from "@tanstack/react-query";
import {
  Clapperboard,
  Cpu,
  Download,
  Film,
  Mic,
  Play,
  Save,
  Settings,
  Workflow,
} from "lucide-react";
import type React from "react";
import { useState } from "react";
import { ApiError, type GpuDevice, fetchGpuStatus, renderVideo } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import { type WorkspacePreset, pipelineNodeHint, pipelineStartForNode } from "../../types/workflow";
import { SettingsPage } from "../SettingsPage";
import { PipelineRunControls } from "../workflow/PipelineRunControls";

const PRESET_OPTIONS: Array<{
  id: WorkspacePreset;
  label: string;
  icon: React.ReactNode;
}> = [
  {
    id: "editing",
    label: "剪辑",
    icon: <Film className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
  {
    id: "workflow",
    label: "节点流",
    icon: <Workflow className="h-3.5 w-3.5" strokeWidth={1.5} />,
  },
];

function formatGpuCompact(gpu: GpuDevice): string {
  return `${(gpu.used_mib / 1024).toFixed(1)}/${(gpu.total_mib / 1024).toFixed(1)}G`;
}

/**
 * 紧凑 36px Premiere 风格顶部主控栏（去文字冗余，图标 + 关键计数优先）。
 */
export const TopMenuBar: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isDirty = useProjectStore((s) => s.isDirty);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);

  const workspacePreset = useWorkflowStore((s) => s.workspacePreset);
  const setWorkspacePreset = useWorkflowStore((s) => s.setWorkspacePreset);
  const pipelineConfig = useWorkflowStore((s) => s.pipelineConfig);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const runPipeline = useWorkflowStore((s) => s.runPipeline);
  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const selectedTitle = useWorkflowStore((s) => s.nodes.find((node) => node.id === s.selectedNodeId)?.data.title);
  const startStep = pipelineStartForNode(selectedNodeId);
  const appendLog = useWorkflowStore((s) => s.appendLog);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [actionToast, setActionToast] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const { data: gpuData } = useQuery({
    queryKey: ["gpu-status"],
    queryFn: ({ signal }) => fetchGpuStatus(signal),
    enabled: backendStatus === "online",
    refetchInterval: 3000,
    retry: false,
  });
  const gpus = gpuData?.devices ?? [];
  const primaryGpu = gpus[0];
  const gpuTitle = gpus.length
    ? gpus.map((gpu) => `${gpu.name} · ${formatGpuCompact(gpu)}`).join("\n")
    : "显存监控不可用";

  const statusColor =
    backendStatus === "online"
      ? "bg-status-ready"
      : backendStatus === "connecting"
        ? "bg-secondary animate-pulse"
        : "bg-status-warning";

  const statusTitle =
    backendStatus === "online"
      ? "后端已连接"
      : backendStatus === "connecting"
        ? "连接中..."
        : "离线模式";

  const pendingCount = script
    ? script.segments.filter((seg) => !seg.keep_original && seg.status !== "ready").length
    : 0;

  const handleBatchSynthesize = async () => {
    if (!script || isSynthesizing) {
      return;
    }
    setActionToast("配音中...");
    const ids =
      pendingCount === 0
        ? script.segments.filter((seg) => !seg.keep_original).map((seg) => seg.id)
        : undefined;
    const res = await synthesizeSegment(ids);
    setActionToast(res.message);
    if (res.issues?.length) {
      res.issues.forEach((issue) => appendLog("env_tts", issue, "warn"));
    } else {
      appendLog("env_tts", res.message, res.simulated ? "warn" : "success");
    }
    window.setTimeout(() => setActionToast(null), 4000);
  };

  const handleExport = async () => {
    if (!script || isExporting) {
      return;
    }
    if (backendStatus === "offline") {
      const msg = "离线模式：已记录导出配置";
      setActionToast(msg);
      appendLog("env_backend", msg, "warn");
      window.setTimeout(() => setActionToast(null), 3000);
      return;
    }
    setIsExporting(true);
    setActionToast("导出中...");
    try {
      if (useProjectStore.getState().isDirty) {
        await saveProjectScript();
        const afterSave = useProjectStore.getState();
        if (afterSave.backendStatus === "offline") {
          const msg = "离线模式：已记录导出配置";
          setActionToast(msg);
          appendLog("env_backend", msg, "warn");
          window.setTimeout(() => setActionToast(null), 3000);
          return;
        }
        if (afterSave.error || afterSave.isDirty) {
          const msg = afterSave.error ?? "请先保存修改";
          setActionToast(msg);
          appendLog("env_backend", msg, "error");
          window.setTimeout(() => setActionToast(null), 3000);
          return;
        }
      }
      const res = await renderVideo(script.project_id, {
        burn_watermark: pipelineConfig.burnWatermark,
        burn_subtitles: pipelineConfig.burnSubtitles,
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
      setActionToast("已导出");
      appendLog("env_backend", `导出成功: ${res.output_video}`, "success");
    } catch (err) {
      const isStub =
        err instanceof ApiError &&
        (err.status === 503 ||
          err.status === 502 ||
          err.status === 504 ||
          err.isNotImplemented ||
          err.isNetworkError);
      const msg = isStub
        ? "离线模式：已记录导出配置"
        : err instanceof Error
          ? err.message
          : "导出失败";
      setActionToast(msg);
      appendLog("env_backend", msg, isStub ? "warn" : "error");
    } finally {
      setIsExporting(false);
    }
    window.setTimeout(() => setActionToast(null), 3000);
  };

  return (
    <header className="flex h-9 shrink-0 items-center justify-between bg-[#222222] px-3 text-[12px]">
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 text-[13px] font-medium text-primary">
          <Clapperboard className="h-4 w-4 text-secondary" strokeWidth={1.5} />
          <span>StageTTS</span>
        </div>

        <span title={statusTitle} className={`h-1.5 w-1.5 shrink-0 rounded-full ${statusColor}`} />

        {script && (
          <>
            <span className="flex h-6 items-center rounded-[4px] bg-white/[0.04] px-2 font-mono tabular-nums text-[11px] text-secondary">
              {script.project_id}
            </span>
            <button
              type="button"
              disabled={isLoading}
              onClick={() => void saveProjectScript()}
              title="保存 (Ctrl+S)"
              aria-label="保存"
              className="relative flex h-7 w-7 items-center justify-center rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
              {isDirty && (
                <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-status-warning" />
              )}
            </button>
          </>
        )}
      </div>

      <nav
        aria-label="工作区模式"
        className="flex items-center gap-0.5 rounded-[4px] bg-[#181818] p-0.5"
      >
        {PRESET_OPTIONS.map((preset) => {
          const active = workspacePreset === preset.id;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => setWorkspacePreset(preset.id)}
              className={`flex h-6 items-center gap-1.5 rounded-[3px] px-3 text-[12px] font-medium transition-colors ${
                active ? "bg-[#333333] text-[#FFFFFF]" : "text-secondary hover:text-primary"
              }`}
            >
              {preset.icon}
              <span>{preset.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="flex items-center gap-1.5">
        {actionToast && (
          <span className="flex h-6 max-w-48 truncate items-center rounded-[4px] bg-white/[0.06] px-2 text-[11px] text-primary">
            {actionToast}
          </span>
        )}

        <div
          title={gpuTitle}
          className="hidden h-7 items-center gap-1 whitespace-nowrap rounded-[4px] bg-white/[0.04] px-2 font-mono tabular-nums text-[11px] text-secondary sm:flex"
        >
          <Cpu className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
          <span>{primaryGpu ? formatGpuCompact(primaryGpu) : "--"}</span>
        </div>

        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          title="设置"
          aria-label="设置"
          className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary hover:bg-white/5 hover:text-primary"
        >
          <Settings className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
        {settingsOpen && <SettingsPage onClose={() => setSettingsOpen(false)} />}

        <PipelineRunControls />

        {workspacePreset === "workflow" ? (
          <button
            type="button"
            disabled={isPipelineRunning || !startStep}
            title={startStep ? `从「${selectedTitle}」运行` : pipelineNodeHint(selectedNodeId)}
            onClick={() => void runPipeline({ parseOnly: false })}
            className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-3 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>{isPipelineRunning ? "运行中" : startStep ? "运行" : "选节点"}</span>
          </button>
        ) : (
          script && (
            <>
              {script.segments.length > 0 && (
                <button
                  type="button"
                  disabled={isSynthesizing || isLoading || isPipelineRunning}
                  onClick={() => void handleBatchSynthesize()}
                  title="批量合成待配音片段"
                  className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.1] bg-[#2C2C2C] px-2.5 text-[12px] font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
                >
                  <Mic className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
                  <span>
                    {isSynthesizing ? "配音中" : pendingCount > 0 ? `配音 (${pendingCount})` : "配音"}
                  </span>
                </button>
              )}
              <button
                type="button"
                disabled={isExporting || isSynthesizing || isLoading || isPipelineRunning}
                onClick={() => void handleExport()}
                title="导出成片"
                className="flex h-7 items-center gap-1.5 whitespace-nowrap rounded-[4px] bg-[#3A6EA5] px-2.5 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
              >
                <Download className="h-3.5 w-3.5" strokeWidth={1.5} />
                <span>{isExporting ? "导出中" : "导出"}</span>
              </button>
            </>
          )
        )}
      </div>
    </header>
  );
};
