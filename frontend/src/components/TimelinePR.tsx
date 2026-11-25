import { AlertCircle, GitMerge, Plus, Scissors, Trash2, ZoomIn, ZoomOut } from "lucide-react";
import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { EmotionType } from "../types/script";
import { normalizeSpeakerColor } from "../utils/speakerColor";
import { isSegmentOverflow, isSegmentTooShort } from "../utils/synthesisIssues";
import { DubbingRange } from "./DubbingRange";

type TimelineFilter = "all" | "review" | "pending" | "issue";

function formatShortTime(sec: number): string {
  const mins = Math.floor(sec / 60);
  const secs = (sec % 60).toFixed(1).padStart(4, "0");
  return `${String(mins).padStart(2, "0")}:${secs}`;
}

const EMOTION_LABELS: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

/**
 * 3 轨 Premiere 风格单色时间轴（5px 短刻度 + 极淡垂直虚线网格 + 纯色块片段）。
 */
export const TimelinePR: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const currentTime = useProjectStore((s) => s.currentTime);
  const dubbingRange = useProjectStore((s) => s.dubbingRange);
  const isPlaying = useProjectStore((s) => s.isPlaying);
  const timelineZoom = useProjectStore((s) => s.timelineZoom);
  const trackStates = useProjectStore((s) => s.trackStates);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const editsDisabled = isLoading || isSynthesizing || pipelineRunning;

  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const setTimelineZoom = useProjectStore((s) => s.setTimelineZoom);
  const toggleTrackState = useProjectStore((s) => s.toggleTrackState);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);
  const commitHistoryCheckpoint = useProjectStore((s) => s.commitHistoryCheckpoint);
  const updateSegmentLocal = useProjectStore((s) => s.updateSegmentLocal);
  const splitSegment = useProjectStore((s) => s.splitSegment);
  const mergeSegmentWithNext = useProjectStore((s) => s.mergeSegmentWithNext);
  const addSegment = useProjectStore((s) => s.addSegment);
  const deleteSegment = useProjectStore((s) => s.deleteSegment);
  const confirmAllPendingReviews = useProjectStore((s) => s.confirmAllPendingReviews);

  const waveContainerRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const activeDragCleanupRef = useRef<(() => void) | null>(null);
  const justDraggedRef = useRef(false);
  const [waveReady, setWaveReady] = useState(false);
  const [segmentFilter, setSegmentFilter] = useState<TimelineFilter>("all");
  const [dragInfo, setDragInfo] = useState<{
    segmentId: number;
    edge: "start" | "end" | "move";
    time: number;
    endTime?: number;
    duration: number;
  } | null>(null);

  const maxSegmentEnd = useMemo(() => {
    if (!script?.segments.length) return 0;
    return Math.max(...script.segments.map((seg) => seg.end));
  }, [script?.segments]);

  const duration = Math.max(10, script?.media.duration ?? 18.5, maxSegmentEnd);
  const trackWidthPx = Math.max(760, Math.ceil(duration * timelineZoom) + 80);

  useEffect(() => {
    const container = waveContainerRef.current;
    const vocalsUrl = script?.media.vocals_wav;
    setWaveReady(false);
    if (!container || !vocalsUrl) {
      return;
    }
    let destroyed = false;
    let ws: WaveSurfer | null = null;
    try {
      ws = WaveSurfer.create({
        container,
        height: 44,
        waveColor: "#6E7681",
        progressColor: "#B0B8C1",
        cursorWidth: 0,
        interact: false,
        normalize: true,
      });
      ws.on("ready", () => {
        if (!destroyed) {
          setWaveReady(true);
        }
      });
      ws.on("error", () => {
        if (!destroyed) {
          setWaveReady(false);
        }
      });
      void ws.load(vocalsUrl).catch(() => {
        if (!destroyed) {
          setWaveReady(false);
        }
      });
    } catch {
      // ignore offline audio fallback
    }
    return () => {
      if (!destroyed && ws) {
        destroyed = true;
        ws.destroy();
      }
    };
  }, [script?.media.vocals_wav]);

  // Ctrl/Cmd/Alt + 滚轮：以指针位置为中心缩放时间轴
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) {
      return;
    }
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey || e.altKey)) {
        return;
      }
      e.preventDefault();
      const state = useProjectStore.getState();
      const oldZoom = state.timelineZoom;
      const delta = e.deltaY < 0 ? 12 : -12;
      const nextZoom = Math.max(30, Math.min(180, Math.round(oldZoom + delta)));
      if (nextZoom === oldZoom) {
        return;
      }
      const rect = el.getBoundingClientRect();
      const offsetInView = Math.max(0, e.clientX - rect.left);
      const timeAtPointer = (el.scrollLeft + offsetInView) / oldZoom;
      state.setTimelineZoom(nextZoom);
      requestAnimationFrame(() => {
        el.scrollLeft = Math.max(0, timeAtPointer * nextZoom - offsetInView);
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // 组件卸载时清理未结束的拖拽窗口事件监听
  useEffect(() => {
    return () => {
      activeDragCleanupRef.current?.();
      activeDragCleanupRef.current = null;
    };
  }, []);

  // 播放时自动跟随播放头进入可视区域
  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    const el = scrollContainerRef.current;
    if (!el) {
      return;
    }
    const playheadX = currentTime * timelineZoom;
    if (playheadX < el.scrollLeft + 24 || playheadX > el.scrollLeft + el.clientWidth - 48) {
      el.scrollLeft = Math.max(0, playheadX - el.clientWidth * 0.3);
    }
  }, [currentTime, isPlaying, timelineZoom]);

  const tickMarks = useMemo(() => {
    const step = timelineZoom >= 90 ? 1 : 2;
    const marks: number[] = [];
    for (let t = 0; t <= Math.ceil(duration); t += step) {
      marks.push(t);
    }
    return marks;
  }, [duration, timelineZoom]);

  const fallbackWaveBars = useMemo(() => {
    const barCount = 120;
    return Array.from({ length: barCount }).map((_, idx) => {
      const t = (idx / barCount) * duration;
      const inSeg = script?.segments.some((seg) => t >= seg.start && t <= seg.end);
      const amp = inSeg
        ? 0.35 + 0.6 * Math.abs(Math.sin(idx * 1.37) * Math.cos(idx * 0.43))
        : 0.08 + 0.08 * Math.abs(Math.sin(idx * 0.7));
      return {
        height: Math.round(amp * 34),
        active: Boolean(inSeg),
      };
    });
  }, [duration, script?.segments]);

  if (!script) {
    return null;
  }

  const pendingCount = script.segments.filter(
    (seg) => !seg.keep_original && seg.status !== "ready",
  ).length;
  const pendingReviewCount = script.segments.filter(
    (seg) =>
      seg.segment_timing_needs_review !== false ||
      seg.alignment_needs_review !== false ||
      (seg.speaker_needs_review !== false && seg.speaker !== "UNKNOWN"),
  ).length;
  const sortedSegments = [...script.segments].sort((a, b) => a.start - b.start || a.id - b.id);
  const selectedSegment = script.segments.find((seg) => seg.id === selectedSegmentId) ?? null;

  const matchesFilter = (seg: (typeof script.segments)[number]) => {
    if (segmentFilter === "all") return true;
    if (segmentFilter === "review") {
      return (
        seg.segment_timing_needs_review !== false ||
        seg.alignment_needs_review !== false ||
        seg.speaker_needs_review !== false
      );
    }
    if (segmentFilter === "pending") {
      return !seg.keep_original && seg.status !== "ready";
    }
    return seg.status === "error" || isSegmentOverflow(seg) || isSegmentTooShort(seg);
  };

  const handleRulerSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const offsetX = Math.max(0, e.clientX - rect.left);
    setCurrentTime(offsetX / timelineZoom);
  };

  const snapToTimelineTargets = (
    raw: number,
    minAllowed: number,
    maxAllowed: number,
    snapTargets: number[],
  ): number => {
    const quantized = Math.round(raw / 0.05) * 0.05;
    const clamped = Math.max(minAllowed, Math.min(maxAllowed, quantized));
    let best = clamped;
    let bestDist = 0.08;
    for (const target of snapTargets) {
      if (target < minAllowed || target > maxAllowed) continue;
      const dist = Math.abs(target - clamped);
      if (dist <= bestDist) {
        bestDist = dist;
        best = target;
      }
    }
    return Number(best.toFixed(2));
  };

  const beginEdgeResize = (
    e: React.PointerEvent<HTMLDivElement>,
    seg: (typeof script.segments)[number],
    edge: "start" | "end",
  ) => {
    e.stopPropagation();
    e.preventDefault();
    if (editsDisabled) {
      return;
    }
    activeDragCleanupRef.current?.();
    const idx = sortedSegments.findIndex((item) => item.id === seg.id);
    const prevEnd = idx > 0 ? sortedSegments[idx - 1].end : 0;
    const nextStart =
      idx >= 0 && idx < sortedSegments.length - 1 ? sortedSegments[idx + 1].start : duration + 30;
    const minDuration = 0.2;
    const originX = e.clientX;
    const initialEdgeTime = edge === "start" ? seg.start : seg.end;
    const minAllowed =
      edge === "start"
        ? Math.min(seg.start, Math.max(0, seg.end - minDuration), Math.max(0, prevEnd))
        : seg.start + minDuration;
    const maxAllowed =
      edge === "start"
        ? Math.max(minAllowed, seg.end - minDuration)
        : Math.max(seg.start + minDuration, seg.end, nextStart);

    selectSegment(seg.id);
    let committed = false;
    setDragInfo({
      segmentId: seg.id,
      edge,
      time: initialEdgeTime,
      duration: Number((seg.end - seg.start).toFixed(2)),
    });

    const onPointerMove = (moveEvent: PointerEvent) => {
      const dx = moveEvent.clientX - originX;
      const rawTime = initialEdgeTime + dx / timelineZoom;
      const snapped = snapToTimelineTargets(rawTime, minAllowed, maxAllowed, [
        currentTime,
        Math.round(rawTime),
        prevEnd,
        nextStart,
      ]);
      if (!committed && snapped !== Number(initialEdgeTime.toFixed(2))) {
        commitHistoryCheckpoint();
        committed = true;
      }
      const nextDuration =
        edge === "start"
          ? Number((seg.end - snapped).toFixed(2))
          : Number((snapped - seg.start).toFixed(2));
      setDragInfo({
        segmentId: seg.id,
        edge,
        time: snapped,
        duration: nextDuration,
      });
      updateSegmentLocal(
        edge === "start" ? { id: seg.id, start: snapped } : { id: seg.id, end: snapped },
        { skipHistory: true },
      );
    };

    const cleanup = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      if (activeDragCleanupRef.current === cleanup) {
        activeDragCleanupRef.current = null;
      }
      setDragInfo(null);
    };

    const onPointerUp = () => {
      if (committed) {
        justDraggedRef.current = true;
        window.setTimeout(() => {
          justDraggedRef.current = false;
        }, 0);
      }
      cleanup();
    };

    activeDragCleanupRef.current = cleanup;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
  };

  const beginClipMove = (
    e: React.PointerEvent<HTMLDivElement>,
    seg: (typeof script.segments)[number],
  ) => {
    if (editsDisabled || e.button !== 0) {
      return;
    }
    activeDragCleanupRef.current?.();
    const idx = sortedSegments.findIndex((item) => item.id === seg.id);
    const prevEnd = idx > 0 ? sortedSegments[idx - 1].end : 0;
    const nextStartBound =
      idx >= 0 && idx < sortedSegments.length - 1 ? sortedSegments[idx + 1].start : duration + 30;
    const clipDuration = Number(Math.max(0.2, seg.end - seg.start).toFixed(2));
    const rawMaxStart = nextStartBound - clipDuration;
    const minStart = Math.max(0, Math.min(seg.start, prevEnd));
    const maxStart = Math.max(minStart, seg.start, rawMaxStart);
    const originX = e.clientX;
    let committed = false;

    const onPointerMove = (moveEvent: PointerEvent) => {
      const dx = moveEvent.clientX - originX;
      if (!committed && Math.abs(dx) < 4) {
        return;
      }
      const rawStart = seg.start + dx / timelineZoom;
      const nextStart = snapToTimelineTargets(rawStart, minStart, maxStart, [
        currentTime,
        Math.round(rawStart),
        prevEnd,
        rawMaxStart,
      ]);
      const nextEnd = Number((nextStart + clipDuration).toFixed(2));
      if (!committed) {
        selectSegment(seg.id);
        commitHistoryCheckpoint();
        committed = true;
      }
      setDragInfo({
        segmentId: seg.id,
        edge: "move",
        time: nextStart,
        endTime: nextEnd,
        duration: clipDuration,
      });
      updateSegmentLocal({ id: seg.id, start: nextStart, end: nextEnd }, { skipHistory: true });
    };

    const cleanup = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      if (activeDragCleanupRef.current === cleanup) {
        activeDragCleanupRef.current = null;
      }
      setDragInfo(null);
    };

    const onPointerUp = () => {
      if (committed) {
        justDraggedRef.current = true;
        window.setTimeout(() => {
          justDraggedRef.current = false;
        }, 0);
      }
      cleanup();
    };

    activeDragCleanupRef.current = cleanup;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
  };

  return (
    <div className="flex h-full select-none flex-col bg-[#222222] text-[12px]">
      {/* 紧凑工具条（右侧留足 pr-4，避免 + 按钮贴边裁切） */}
      <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-b border-white/[0.06] bg-[#222222] pl-3.5 pr-4">
        <div className="flex min-w-0 items-center gap-2">
          <span className="shrink-0 text-[13px] font-medium text-primary">时间轴</span>
          <span className="truncate font-mono tabular-nums text-[11px] text-secondary">
            <span className="text-primary">{formatShortTime(currentTime)}</span> /{" "}
            {formatShortTime(duration)}
          </span>
          {dragInfo && (
            <span className="shrink-0 rounded-[3px] bg-[#181818] px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-primary">
              #{dragInfo.segmentId}{" "}
              {dragInfo.edge === "move"
                ? `移动 ${dragInfo.time.toFixed(2)}s–${(dragInfo.endTime ?? dragInfo.time).toFixed(2)}s`
                : `${dragInfo.edge === "start" ? "入点" : "出点"} ${dragInfo.time.toFixed(2)}s`}{" "}
              · {dragInfo.duration.toFixed(2)}s
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <div className="hidden items-center gap-0.5 rounded-[4px] bg-[#181818] p-0.5 text-[10px] lg:flex">
            {(
              [
                { id: "all", label: "全部" },
                { id: "review", label: "待校对" },
                { id: "pending", label: "待合成" },
                { id: "issue", label: "异常" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setSegmentFilter(tab.id)}
                className={`rounded-[3px] px-1.5 py-0.5 transition-colors ${
                  segmentFilter === tab.id
                    ? "bg-[#333333] text-[#FFFFFF]"
                    : "text-secondary hover:text-primary"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {pendingReviewCount > 0 && (
            <button
              type="button"
              disabled={editsDisabled}
              onClick={() => confirmAllPendingReviews()}
              title="一键确认全部待校对片段的边界、对齐与角色"
              className="flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.08] bg-[#282828] px-2 py-1 text-[11px] text-primary transition-colors hover:bg-[#343434] disabled:opacity-45"
            >
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-[#D4933D]" />
              <span className="font-mono tabular-nums">全部确认 ({pendingReviewCount})</span>
            </button>
          )}

          <div className="flex items-center gap-1 rounded-[4px] bg-[#181818] p-0.5">
            <button
              type="button"
              disabled={editsDisabled}
              onClick={() => addSegment()}
              title="在播放头位置新增片段"
              className="flex h-6 items-center gap-1 rounded-[3px] px-1.5 text-[11px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-35"
            >
              <Plus className="h-3 w-3" strokeWidth={1.5} />
              <span className="hidden sm:inline">新增</span>
            </button>
            <button
              type="button"
              disabled={
                editsDisabled ||
                !selectedSegment ||
                selectedSegment.end - selectedSegment.start < 0.4
              }
              onClick={() => selectedSegment && splitSegment(selectedSegment.id, currentTime)}
              title="分割选中片段（按播放头位置或中点）"
              className="flex h-6 items-center gap-1 rounded-[3px] px-1.5 text-[11px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-35"
            >
              <Scissors className="h-3 w-3" strokeWidth={1.5} />
              <span className="hidden sm:inline">分割</span>
            </button>
            <button
              type="button"
              disabled={editsDisabled || !selectedSegment}
              onClick={() => selectedSegment && mergeSegmentWithNext(selectedSegment.id)}
              title="与下一个片段合并"
              className="flex h-6 items-center gap-1 rounded-[3px] px-1.5 text-[11px] text-secondary transition-colors hover:bg-[#333333] hover:text-primary disabled:opacity-35"
            >
              <GitMerge className="h-3 w-3" strokeWidth={1.5} />
              <span className="hidden md:inline">合并</span>
            </button>
            <button
              type="button"
              disabled={editsDisabled || !selectedSegment}
              onClick={() => selectedSegment && deleteSegment(selectedSegment.id)}
              title="删除选中片段"
              className="flex h-6 items-center gap-1 rounded-[3px] px-1.5 text-[11px] text-secondary transition-colors hover:bg-[#333333] hover:text-status-error disabled:opacity-35"
            >
              <Trash2 className="h-3 w-3" strokeWidth={1.5} />
            </button>
          </div>

          {/* 次操作按钮：#2C2C2C 灰底 + 极淡边框 + 6px #D4933D 琥珀色圆点 */}
          <button
            type="button"
            onClick={() => void synthesizeSegment()}
            disabled={isSynthesizing}
            className="flex h-7 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2.5 py-1 text-[11px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
          >
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full"
              style={{
                backgroundColor: pendingCount > 0 ? "#D4933D" : "#4FA882",
              }}
            />
            <span className="font-mono tabular-nums">
              {isSynthesizing
                ? "合成中..."
                : pendingCount > 0
                  ? `合成待处理 (${pendingCount})`
                  : "重合成待处理"}
            </span>
          </button>

          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={() => setTimelineZoom(timelineZoom - 15)}
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
              title="缩小"
            >
              <ZoomOut className="h-3.5 w-3.5" strokeWidth={1.5} />
            </button>
            <input
              aria-label="时间轴缩放"
              type="range"
              min={30}
              max={180}
              step={6}
              value={timelineZoom}
              onChange={(e) => setTimelineZoom(Number(e.target.value))}
              className="w-20 shrink-0"
            />
            <button
              type="button"
              onClick={() => setTimelineZoom(timelineZoom + 15)}
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
              title="放大"
            >
              <ZoomIn className="h-3.5 w-3.5" strokeWidth={1.5} />
            </button>
          </div>
        </div>
      </div>

      {/* 3 轨主体 */}
      <div className="flex min-h-0 flex-1 overflow-y-auto overflow-x-hidden bg-[#161616]">
        {/* 左侧固定轨道头 (104px)，#222222 哑光灰，轨道间极弱 5% 分隔线 */}
        <div className="w-[104px] shrink-0 border-r border-white/[0.06] bg-[#222222]">
          {/* 标尺占位 (24px) */}
          <div className="h-6 border-b border-white/[0.05]" />

          {/* 视频 V1 (32px) */}
          <div className="flex h-8 items-center justify-between border-b border-white/[0.05] px-2.5">
            <span className="whitespace-nowrap text-[11px] font-medium text-secondary">
              视频 V1
            </span>
            <button
              type="button"
              onClick={() => toggleTrackState("videoMuted")}
              className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                trackStates.videoMuted
                  ? "bg-[#2C2C2C] text-status-error"
                  : "text-muted hover:text-primary"
              }`}
            >
              M
            </button>
          </div>

          {/* 配音 A1 (72px) */}
          <div className="flex h-[72px] items-center justify-between border-b border-white/[0.05] px-2.5">
            <span className="whitespace-nowrap text-[11px] font-medium text-secondary">
              配音 A1
            </span>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => toggleTrackState("dubMuted")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.dubMuted
                    ? "bg-[#2C2C2C] text-status-error"
                    : "text-muted hover:text-primary"
                }`}
              >
                M
              </button>
              <button
                type="button"
                onClick={() => toggleTrackState("dubSolo")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.dubSolo
                    ? "bg-[#2C2C2C] text-status-warning"
                    : "text-muted hover:text-primary"
                }`}
              >
                S
              </button>
            </div>
          </div>

          {/* 波形 / BGM A2 (56px) */}
          <div className="flex h-14 items-center justify-between px-2.5">
            <span className="truncate whitespace-nowrap text-[11px] font-medium text-secondary">
              原声/BGM
            </span>
            <div className="flex shrink-0 gap-1">
              <button
                type="button"
                title="静音原声"
                onClick={() => toggleTrackState("vocalsMuted")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.vocalsMuted
                    ? "bg-[#2C2C2C] text-status-error"
                    : "text-muted hover:text-primary"
                }`}
              >
                M
              </button>
              <button
                type="button"
                title="静音伴奏 (BGM)"
                onClick={() => toggleTrackState("bgmMuted")}
                className={`rounded-[3px] px-1 font-mono text-[10px] font-medium transition-colors ${
                  trackStates.bgmMuted
                    ? "bg-[#2C2C2C] text-status-error"
                    : "text-muted hover:text-primary"
                }`}
              >
                B
              </button>
            </div>
          </div>
        </div>

        {/* 右侧可滚动轨道画布（#161616 深灰井） */}
        <div
          ref={scrollContainerRef}
          className="relative flex-1 overflow-x-auto overflow-y-hidden bg-[#161616] pl-2.5"
        >
          <div className="relative min-h-full" style={{ width: `${trackWidthPx}px` }}>
            {/* 贯穿所有轨道的极淡垂直虚线时间网格（rgba(255,255,255,0.05)） */}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 top-6 z-0">
              {tickMarks.map((sec) => (
                <div
                  key={`grid-${sec}`}
                  style={{ left: `${sec * timelineZoom}px` }}
                  className="absolute bottom-0 top-0 border-l border-dashed border-white/[0.05]"
                />
              ))}
            </div>

            {dubbingRange && (
              <div
                style={{
                  left: `${dubbingRange.start * timelineZoom}px`,
                  width: `${Math.max(2, (dubbingRange.end - dubbingRange.start) * timelineZoom)}px`,
                }}
                className="pointer-events-none absolute bottom-0 top-6 z-[5] border-x border-[#4FA882]/45 bg-[#4FA882]/[0.07]"
              />
            )}

            {/* 时间标尺 (24px)：底部 5px 小短刻度，彻底去掉亮色方框 */}
            <div
              onClick={handleRulerSeek}
              className="relative z-10 h-6 cursor-pointer border-b border-white/[0.05] bg-[#1C1C1C]"
            >
              {tickMarks.map((sec) => (
                <div
                  key={sec}
                  style={{ left: `${sec * timelineZoom}px` }}
                  className="absolute bottom-0 top-0"
                >
                  <span className="block pl-1 pt-0.5 font-mono tabular-nums text-[10px] leading-none text-muted">
                    {sec}s
                  </span>
                  <span className="absolute bottom-0 left-0 h-[5px] w-px bg-white/[0.15]" />
                </div>
              ))}
            </div>

            {/* V1 (32px)：纯色块 #2A2A2A，无外框 */}
            <div className="relative z-10 h-8 border-b border-white/[0.05] py-1">
              <div
                style={{ width: `${duration * timelineZoom}px` }}
                className={`flex h-full items-center rounded-[4px] bg-[#2A2A2A] px-2.5 text-[11px] text-secondary ${
                  trackStates.videoMuted ? "opacity-40" : ""
                }`}
              >
                <span className="truncate font-mono tabular-nums">
                  源画面轨 · {duration.toFixed(1)}s
                </span>
              </div>
            </div>

            {/* A1 配音块轨 (72px)：纯色块 #2A2A2A + 仅左侧 3px 角色色条；选中才显示 1px 低饱和角色色外框 */}
            <div
              className={`relative z-10 h-[72px] border-b border-white/[0.05] ${
                trackStates.dubMuted ? "opacity-40" : ""
              }`}
            >
              {script.segments.map((seg) => {
                const character = script.characters[seg.speaker];
                const speakerAccent = normalizeSpeakerColor(character?.color);
                const slotDuration = Math.max(0.2, seg.end - seg.start);
                const left = seg.start * timelineZoom;
                const width = Math.max(6, slotDuration * timelineZoom);
                const isSelected = seg.id === selectedSegmentId;
                const isOverflow = isSegmentOverflow(seg);
                const isDimmed = !matchesFilter(seg);
                const tooltip = `#${seg.id} ${character?.display_name ?? seg.speaker} (${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s)\n${seg.target_text}`;

                return (
                  <div
                    key={seg.id}
                    role="button"
                    tabIndex={0}
                    draggable={false}
                    title={`${tooltip}\n提示：左右拖动块体平移片段，拖动左右边缘调整入/出点`}
                    onPointerDown={(e) => beginClipMove(e, seg)}
                    onClick={() => {
                      if (justDraggedRef.current) {
                        return;
                      }
                      selectSegment(seg.id);
                      setCurrentTime(seg.start);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        selectSegment(seg.id);
                        setCurrentTime(seg.start);
                      }
                    }}
                    style={{
                      left: `${left}px`,
                      width: `${width}px`,
                      borderColor: isSelected ? speakerAccent : "transparent",
                      borderLeftColor: speakerAccent,
                    }}
                    className={`group absolute inset-y-1.5 flex touch-none flex-col justify-between overflow-hidden rounded-[4px] border border-l-[3px] text-left transition-colors ${
                      editsDisabled ? "cursor-default" : "cursor-grab active:cursor-grabbing"
                    } ${
                      width >= 76 ? "px-2 py-1" : width >= 32 ? "px-1 py-1" : "px-0.5 py-1"
                    } ${
                      isSelected ? "z-10 bg-[#333333]" : "bg-[#2A2A2A] hover:bg-[#303030]"
                    } ${isDimmed ? "opacity-25" : ""}`}
                  >
                    <div
                      onPointerDown={(e) => beginEdgeResize(e, seg, "start")}
                      title="拖拽调整入点"
                      className="absolute inset-y-0 left-0 z-20 w-2 cursor-ew-resize bg-transparent transition-colors group-hover:bg-white/[0.12]"
                    />
                    <div
                      onPointerDown={(e) => beginEdgeResize(e, seg, "end")}
                      title="拖拽调整出点"
                      className="absolute inset-y-0 right-0 z-20 w-2 cursor-ew-resize bg-transparent transition-colors group-hover:bg-white/[0.12]"
                    />

                    {width < 32 ? (
                      <div className="flex h-full items-center justify-center">
                        <span
                          className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                            seg.status === "ready"
                              ? "bg-status-ready"
                              : seg.status === "pending"
                                ? "bg-status-warning"
                                : "bg-status-error"
                          }`}
                        />
                      </div>
                    ) : width < 76 ? (
                      <>
                        <div className="flex items-center justify-between gap-0.5 text-[10px] leading-tight">
                          <span className="truncate font-mono tabular-nums text-secondary">
                            #{seg.id}
                          </span>
                          <span
                            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                              seg.status === "ready"
                                ? "bg-status-ready"
                                : seg.status === "pending"
                                  ? "bg-status-warning"
                                  : "bg-status-error"
                            }`}
                          />
                        </div>
                        <p className="truncate text-[10px] leading-tight text-primary">
                          {seg.target_text}
                        </p>
                      </>
                    ) : (
                      <>
                        <div className="flex items-center justify-between gap-1 text-[10px] leading-tight">
                          <span className="truncate font-medium text-secondary">
                            <span className="font-mono tabular-nums">#{seg.id}</span>{" "}
                            {character?.display_name ?? seg.speaker}
                          </span>
                          <span
                            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                              seg.status === "ready"
                                ? "bg-status-ready"
                                : seg.status === "pending"
                                  ? "bg-status-warning"
                                  : "bg-status-error"
                            }`}
                          />
                        </div>

                        <p className="truncate text-[11px] leading-tight text-primary">
                          {seg.target_text}
                        </p>

                        <div className="flex items-center gap-1 text-[10px] leading-none">
                          <span className="rounded-[3px] bg-black/25 px-1 py-0.5 text-secondary">
                            {EMOTION_LABELS[seg.emotion]}
                          </span>
                          {seg.keep_original && (
                            <span className="rounded-[3px] bg-black/25 px-1 py-0.5 text-status-ready">
                              原声
                            </span>
                          )}
                          {isOverflow && (
                            <span className="flex items-center gap-0.5 rounded-[3px] bg-black/25 px-1 py-0.5 text-status-error">
                              <AlertCircle className="h-2.5 w-2.5" strokeWidth={1.5} />
                              超时
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {/* A2 中性波形与 BGM 包络轨 (56px) */}
            <div
              className={`relative z-10 flex h-14 items-center ${
                trackStates.vocalsMuted ? "opacity-35" : ""
              }`}
            >
              <div
                ref={waveContainerRef}
                className="pointer-events-none absolute inset-0 opacity-85"
                style={{ width: `${duration * timelineZoom}px` }}
              />
              {!waveReady && (
                <div
                  style={{ width: `${duration * timelineZoom}px` }}
                  className="pointer-events-none flex h-full items-center justify-between px-1"
                >
                  {fallbackWaveBars.map((bar, i) => (
                    <div
                      key={`wave-bar-${i}-${bar.height}`}
                      className={`w-[2px] rounded-[1px] ${
                        bar.active ? "bg-[#B0B8C1]/65" : "bg-[#6E7681]/50"
                      }`}
                      style={{ height: `${bar.height}px` }}
                    />
                  ))}
                </div>
              )}
              {/* 极淡 BGM 增益参考虚线 */}
              <div
                className="pointer-events-none absolute left-0 border-t border-dashed border-white/[0.14]"
                style={{
                  width: `${duration * timelineZoom}px`,
                  bottom: `${Math.min(88, Math.max(12, script.media.bgm_volume * 45))}%`,
                }}
              />
              <span className="pointer-events-none absolute right-2 top-1 rounded-[4px] bg-black/60 px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-secondary">
                伴奏 {Math.round(script.media.bgm_volume * 100)}%
              </span>
            </div>

            {/* 1.5px 珊瑚红播放头竖线（贯穿时间标尺与所有轨道） */}
            <div
              style={{ left: `${currentTime * timelineZoom}px` }}
              className="pointer-events-none absolute bottom-0 top-0 z-20 w-[1.5px] bg-[#D94A4A]"
            >
              <div className="-ml-[3.75px] h-2 w-[9px] rounded-b-[1px] bg-[#D94A4A]" />
            </div>
          </div>
        </div>
      </div>
      <DubbingRange />
    </div>
  );
};
