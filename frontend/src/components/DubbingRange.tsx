import { Play, ScissorsLineDashed, Sparkles, X } from "lucide-react";
import type React from "react";
import { useMemo } from "react";
import { useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";

function formatClock(sec: number): string {
  const safe = Math.max(0, sec);
  const mins = Math.floor(safe / 60);
  const rem = (safe % 60).toFixed(1).padStart(4, "0");
  return `${String(mins).padStart(2, "0")}:${rem}`;
}

export const DubbingRange: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const currentTime = useProjectStore((s) => s.currentTime);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const dubbingRange = useProjectStore((s) => s.dubbingRange);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const gpuBusy = useProjectStore((s) => s.gpuBusy);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);

  const setDubbingRange = useProjectStore((s) => s.setDubbingRange);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const synthesizeRange = useProjectStore((s) => s.synthesizeRange);

  const disabled = isLoading || isSynthesizing || gpuBusy || pipelineRunning;

  const duration = useMemo(() => {
    if (!script) return 30;
    const maxSegEnd = script.segments.reduce((acc, seg) => Math.max(acc, seg.end), 0);
    return Math.max(1, script.media.duration ?? 30, maxSegEnd);
  }, [script]);

  const selectedSegment = useMemo(
    () => script?.segments.find((seg) => seg.id === selectedSegmentId) ?? null,
    [script, selectedSegmentId],
  );

  const rangeSegmentIds = useMemo(() => {
    if (!script || !dubbingRange) return [];
    return script.segments
      .filter(
        (seg) =>
          !seg.keep_original &&
          seg.end > dubbingRange.start + 0.05 &&
          seg.start < dubbingRange.end - 0.05,
      )
      .map((seg) => seg.id);
  }, [script, dubbingRange]);

  if (!script) {
    return null;
  }

  const applyPreset = (start: number, end: number) => {
    const safeStart = Math.max(0, Math.min(duration - 0.2, start));
    const safeEnd = Math.min(duration, Math.max(safeStart + 0.2, end));
    setDubbingRange({
      start: Number(safeStart.toFixed(2)),
      end: Number(safeEnd.toFixed(2)),
    });
  };

  return (
    <div className="flex min-h-9 flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] bg-[#1F1F1F] px-3 py-1.5 text-[11px]">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="flex items-center gap-1 font-medium text-secondary">
          <ScissorsLineDashed className="h-3.5 w-3.5 text-[#4FA882]" strokeWidth={1.5} />
          配音选区
        </span>

        {!dubbingRange ? (
          <>
            <button
              type="button"
              disabled={disabled}
              onClick={() => applyPreset(0, Math.min(duration, 30))}
              className="rounded-[3px] bg-[#2C2C2C] px-2 py-0.5 text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
            >
              前 30s
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => applyPreset(0, Math.min(duration, 60))}
              className="rounded-[3px] bg-[#2C2C2C] px-2 py-0.5 text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
            >
              前 60s
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => applyPreset(currentTime, Math.min(duration, currentTime + 20))}
              className="rounded-[3px] bg-[#2C2C2C] px-2 py-0.5 text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
            >
              从播放头 +20s
            </button>
            {selectedSegment && (
              <button
                type="button"
                disabled={disabled}
                onClick={() => applyPreset(selectedSegment.start, selectedSegment.end)}
                className="rounded-[3px] bg-[#2C2C2C] px-2 py-0.5 text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
              >
                当前片段
              </button>
            )}
          </>
        ) : (
          <>
            <label className="flex items-center gap-1 text-muted">
              起
              <input
                aria-label="选区开始时间"
                type="number"
                min={0}
                max={Math.max(0, dubbingRange.end - 0.2)}
                step={0.5}
                value={dubbingRange.start}
                disabled={disabled}
                onChange={(e) => {
                  const nextStart = Number(e.target.value);
                  if (Number.isFinite(nextStart)) {
                    applyPreset(nextStart, dubbingRange.end);
                  }
                }}
                className="h-6 w-16 rounded-[3px] border border-white/[0.08] bg-[#161616] px-1.5 font-mono tabular-nums text-primary outline-none focus:border-[#4FA882]"
              />
            </label>
            <span className="text-muted">–</span>
            <label className="flex items-center gap-1 text-muted">
              止
              <input
                aria-label="选区结束时间"
                type="number"
                min={Math.min(duration, dubbingRange.start + 0.2)}
                max={duration}
                step={0.5}
                value={dubbingRange.end}
                disabled={disabled}
                onChange={(e) => {
                  const nextEnd = Number(e.target.value);
                  if (Number.isFinite(nextEnd)) {
                    applyPreset(dubbingRange.start, nextEnd);
                  }
                }}
                className="h-6 w-16 rounded-[3px] border border-white/[0.08] bg-[#161616] px-1.5 font-mono tabular-nums text-primary outline-none focus:border-[#4FA882]"
              />
            </label>
            <span className="font-mono tabular-nums text-muted">
              ({formatClock(dubbingRange.start)}–{formatClock(dubbingRange.end)} · {rangeSegmentIds.length} 句)
            </span>
          </>
        )}
      </div>

      {dubbingRange && (
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setCurrentTime(dubbingRange.start)}
            title="跳转到选区起点"
            className="flex h-6 items-center gap-1 rounded-[3px] bg-[#2C2C2C] px-2 text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
          >
            <Play className="h-3 w-3" strokeWidth={1.5} />
            跳至起点
          </button>
          <button
            type="button"
            disabled={disabled || rangeSegmentIds.length === 0}
            onClick={() => void synthesizeRange()}
            className="flex h-6 items-center gap-1 rounded-[3px] border border-[#4FA882]/40 bg-[#4FA882]/15 px-2.5 font-medium text-[#7AD6AE] transition-colors hover:bg-[#4FA882]/25 disabled:opacity-40"
          >
            <Sparkles className="h-3 w-3" strokeWidth={1.5} />
            <span className="font-mono tabular-nums">
              {isSynthesizing ? "合成中..." : `合成选区 (${rangeSegmentIds.length})`}
            </span>
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setDubbingRange(null)}
            title="清除选区"
            className="flex h-6 items-center gap-0.5 rounded-[3px] px-1.5 text-muted transition-colors hover:bg-[#2C2C2C] hover:text-primary disabled:opacity-40"
          >
            <X className="h-3 w-3" strokeWidth={1.5} />
            清除
          </button>
        </div>
      )}
    </div>
  );
};
