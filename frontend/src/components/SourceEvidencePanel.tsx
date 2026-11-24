import { useState } from "react";
import { analyzePerformance } from "../services/api";
import { useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { ProjectScript, Segment } from "../types/script";

const STATUS_LABELS: Record<string, string> = {
  ok: "已分析",
  partial: "部分完成",
  not_run: "未运行",
  not_configured: "未配置",
  blocked: "待复核",
  error: "失败",
};

export function SourceEvidencePanel({ script, segment }: { script: ProjectScript; segment: Segment }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const dirty = useProjectStore((state) => state.isDirty);
  const online = useProjectStore((state) => state.backendStatus === "online");
  const loading = useProjectStore((state) => state.isLoading || state.isSynthesizing);
  const pipelineRunning = useWorkflowStore((state) => state.isPipelineRunning);
  const setScript = useProjectStore((state) => state.setScript);
  const evidence = script.evidence;
  const annotations =
    evidence?.annotations?.filter(
      (item) =>
        (segment.source_turn_ids ?? []).some((id) => (item.source_turn_ids ?? []).includes(id)) ||
        (segment.source_token_ids ?? []).some((id) => (item.source_token_ids ?? []).includes(id)),
    ) ?? [];

  async function analyze(emotion: boolean) {
    setBusy(true);
    setNote(null);
    try {
      const result = await analyzePerformance(script.project_id, [segment.id], emotion);
      const current = useProjectStore.getState();
      if (current.script?.project_id === script.project_id && !current.isDirty) setScript(result.script);
      setNote(
        [STATUS_LABELS[result.status] ?? result.status, ...(result.errors ?? []), ...(result.review_reasons ?? [])].join(" · "),
      );
    } catch (error) {
      setNote(error instanceof Error ? error.message : "分析失败");
    } finally {
      setBusy(false);
    }
  }

  const disabled = busy || dirty || !online || loading || pipelineRunning;
  return (
    <details className="border-t border-dashed border-white/[0.08] pt-2.5 text-[11px]">
      <summary className="flex cursor-pointer select-none items-center justify-between text-secondary hover:text-primary">
        <span>声学分析</span>
        {annotations.length > 0 && (
          <span className="font-mono text-[10px] text-muted">{annotations.length}</span>
        )}
      </summary>
      <div className="mt-2 space-y-2">
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            disabled={disabled}
            title={dirty ? "请先保存修改" : "提取响度/基频韵律"}
            className="rounded border border-white/[0.1] bg-[#282828] px-2 py-1 text-primary hover:bg-[#323232] disabled:opacity-40"
            onClick={() => void analyze(false)}
          >
            韵律
          </button>
          <button
            type="button"
            disabled={disabled}
            title={dirty ? "请先保存修改" : "提取 emotion2vec+ 与韵律"}
            className="rounded border border-white/[0.1] bg-[#282828] px-2 py-1 text-primary hover:bg-[#323232] disabled:opacity-40"
            onClick={() => void analyze(true)}
          >
            情绪+韵律
          </button>
        </div>
        {annotations.map((annotation) => (
          <div key={annotation.id} className="space-y-1 rounded border border-white/[0.07] bg-[#181818] p-2 text-secondary">
            <div className="font-mono text-[10px] text-muted">
              {annotation.start.toFixed(2)}–{annotation.end.toFixed(2)}s ·{" "}
              {annotation.status ? (STATUS_LABELS[annotation.status] ?? annotation.status) : "—"}
            </div>
            {(annotation.emotions ?? []).length > 0 && (
              <div>
                {(annotation.emotions ?? [])
                  .map((item) => `${item.label}${item.score == null ? "" : ` ${item.score.toFixed(2)}`}`)
                  .join(" / ")}
              </div>
            )}
            <div className="flex flex-wrap gap-x-2 text-[10px]">
              {(annotation.prosody ?? [])
                .filter((item) => item.value != null)
                .map((item) => (
                  <span key={item.name}>
                    {item.name}: {item.value?.toFixed(2)} {item.unit}
                  </span>
                ))}
            </div>
          </div>
        ))}
        {note && (
          <p role="status" className="text-status-warning">
            {note}
          </p>
        )}
      </div>
    </details>
  );
}
