import {
  ChevronLeft,
  ChevronRight,
  GitMerge,
  Mic,
  Pause,
  Play,
  Save,
  Scissors,
  Trash2,
  Volume2,
} from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { getCacheBustedUrl, useProjectStore } from "../store/useProjectStore";
import { useWorkflowStore } from "../store/useWorkflowStore";
import type { EmotionType } from "../types/script";
import { isSegmentOverflow, isSegmentTooShort } from "../utils/synthesisIssues";
import { SourceEvidencePanel } from "./SourceEvidencePanel";

const EMOTIONS: Array<{ id: EmotionType; short: string; label: string; dotColor: string }> = [
  { id: "neutral", short: "平", label: "平淡", dotColor: "#9C9C9C" },
  { id: "angry", short: "怒", label: "愤怒", dotColor: "#C95252" },
  { id: "happy", short: "喜", label: "喜悦", dotColor: "#4FA882" },
  { id: "sad", short: "悲", label: "悲伤", dotColor: "#5B8DEF" },
  { id: "whisper", short: "低", label: "低语", dotColor: "#9A86B8" },
];

function formatMiniTime(sec: number): string {
  const safe = Math.max(0, Number.isFinite(sec) ? sec : 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/**
 * Premiere 风格紧凑精修面板（配音主控置顶，原文与校对折叠收纳）。
 */
export const Inspector: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const selectedSegmentId = useProjectStore((s) => s.selectedSegmentId);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const isLoading = useProjectStore((s) => s.isLoading);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const audioVersionMap = useProjectStore((s) => s.audioVersionMap);

  const stepSegment = useProjectStore((s) => s.stepSegment);
  const updateSegmentLocal = useProjectStore((s) => s.updateSegmentLocal);
  const splitSegment = useProjectStore((s) => s.splitSegment);
  const mergeSegmentWithNext = useProjectStore((s) => s.mergeSegmentWithNext);
  const deleteSegment = useProjectStore((s) => s.deleteSegment);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const synthesizeSegment = useProjectStore((s) => s.synthesizeSegment);
  const currentTime = useProjectStore((s) => s.currentTime);

  const [synthNote, setSynthNote] = useState<string | null>(null);
  const [startDraft, setStartDraft] = useState("");
  const [endDraft, setEndDraft] = useState("");
  const [sourcePlaying, setSourcePlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const sourceAudioRef = useRef<HTMLAudioElement>(null);
  const [isAuditionPlaying, setIsAuditionPlaying] = useState(false);
  const [auditionTime, setAuditionTime] = useState(0);
  const [auditionDuration, setAuditionDuration] = useState(0);

  const segment = script?.segments.find((seg) => seg.id === selectedSegmentId) ?? null;
  const auditionUrl = segment
    ? getCacheBustedUrl(segment.tts_wav, audioVersionMap[segment.id])
    : null;
  const fallbackDuration = segment
    ? Math.max(0.1, segment.actual_duration ?? segment.end - segment.start)
    : 0;
  const activeSegmentId = segment?.id ?? null;

  useEffect(() => {
    if (activeSegmentId === null && auditionUrl === null) {
      return;
    }
    const el = audioRef.current;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    setIsAuditionPlaying(false);
    setAuditionTime(0);
    setAuditionDuration(fallbackDuration);
  }, [activeSegmentId, auditionUrl, fallbackDuration]);

  useEffect(() => {
    if (activeSegmentId !== null) {
      sourceAudioRef.current?.pause();
      setSourcePlaying(false);
    }
  }, [activeSegmentId]);

  const segStart = segment?.start;
  const segEnd = segment?.end;
  useEffect(() => {
    if (activeSegmentId !== null && segStart !== undefined && segEnd !== undefined) {
      setStartDraft(segStart.toFixed(2));
      setEndDraft(segEnd.toFixed(2));
    }
  }, [activeSegmentId, segStart, segEnd]);

  if (!script || !segment) {
    return (
      <div className="flex h-full items-center justify-center bg-[#222222] p-4 text-[12px] text-muted">
        点击时间轴片段
      </div>
    );
  }

  const speakers = Object.keys(script.characters);
  const editsDisabled = isLoading || isSynthesizing || pipelineRunning;
  const slotDuration = Math.max(0.1, segment.end - segment.start);
  const targetLang = segment.target_lang.replace(/^all_/, "").split("-")[0];
  const isCjk = ["zh", "ja", "ko", "yue"].includes(targetLang);
  const unit = isCjk ? "字" : "词";
  const usedUnits = isCjk
    ? Array.from(segment.target_text.replace(/\s+/g, "")).length
    : segment.target_text.trim().split(/\s+/).filter(Boolean).length;
  const recommendedUnits = Math.max(1, Math.floor(slotDuration * (isCjk ? 5 : 2.8)));
  const actualDuration = segment.actual_duration;
  const fillRatio = actualDuration ? actualDuration / slotDuration : 0;
  const isOverflow = isSegmentOverflow(segment);
  const isTooShort = isSegmentTooShort(segment);
  const limitRatio = segment.applied_speed_factor == null ? 1.15 : 1.0;

  const effectiveDuration = auditionDuration > 0 ? auditionDuration : fallbackDuration;
  const auditionProgress =
    effectiveDuration > 0
      ? Math.min(100, Math.max(0, (auditionTime / effectiveDuration) * 100))
      : 0;

  const toggleAuditionPlay = async () => {
    const el = audioRef.current;
    if (!el || !auditionUrl) return;
    if (!el.paused) {
      el.pause();
      setIsAuditionPlaying(false);
      return;
    }
    try {
      if (el.ended) el.currentTime = 0;
      await el.play();
      if (audioRef.current === el && !el.paused) setIsAuditionPlaying(true);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setSynthNote("播放失败");
      }
    }
  };

  const handleAuditionSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const nextSec = ratio * effectiveDuration;
    setAuditionTime(nextSec);
    if (audioRef.current && Number.isFinite(audioRef.current.duration)) {
      audioRef.current.currentTime = nextSec;
    }
  };

  const handleSingleSynthesize = async () => {
    const res = await synthesizeSegment([segment.id]);
    setSynthNote(res.message);
    window.setTimeout(() => setSynthNote(null), 3500);
  };

  const commitStartDraft = () => {
    const trimmed = startDraft.trim();
    const val = Number(trimmed);
    if (!trimmed || !Number.isFinite(val) || val < 0 || segment.end - val < 0.2) {
      setStartDraft(segment.start.toFixed(2));
      return;
    }
    const rounded = Number(val.toFixed(2));
    setStartDraft(rounded.toFixed(2));
    if (rounded !== Number(segment.start.toFixed(2))) {
      updateSegmentLocal({ id: segment.id, start: rounded });
    }
  };

  const commitEndDraft = () => {
    const trimmed = endDraft.trim();
    const val = Number(trimmed);
    if (!trimmed || !Number.isFinite(val) || val - segment.start < 0.2) {
      setEndDraft(segment.end.toFixed(2));
      return;
    }
    const rounded = Number(val.toFixed(2));
    setEndDraft(rounded.toFixed(2));
    if (rounded !== Number(segment.end.toFixed(2))) {
      updateSegmentLocal({ id: segment.id, end: rounded });
    }
  };

  const hasPendingReview =
    segment.segment_timing_needs_review !== false ||
    segment.alignment_needs_review !== false ||
    segment.speaker_needs_review !== false;

  return (
    <div className="flex h-full flex-col justify-between overflow-hidden bg-[#222222] text-[12px]">
      <div className="flex-1 space-y-3 overflow-y-auto py-3 pl-3.5 pr-4">
        {/* 片段定位与边界时间 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-baseline gap-2 truncate">
              <span className="font-mono text-[13px] font-medium text-primary">#{segment.id}</span>
              <span className="font-mono tabular-nums text-[11px] text-secondary">
                {slotDuration.toFixed(2)}s
              </span>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <button
                type="button"
                disabled={editsDisabled || slotDuration < 0.4}
                onClick={() => splitSegment(segment.id, currentTime)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
                title="分割片段"
              >
                <Scissors className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                disabled={editsDisabled}
                onClick={() => mergeSegmentWithNext(segment.id)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary disabled:opacity-40"
                title="与下一句合并"
              >
                <GitMerge className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                disabled={editsDisabled}
                onClick={() => deleteSegment(segment.id)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-status-error disabled:opacity-40"
                title="删除片段"
              >
                <Trash2 className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                onClick={() => stepSegment(-1)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
                title="上一句 (←)"
              >
                <ChevronLeft className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
              <button
                type="button"
                onClick={() => stepSegment(1)}
                className="flex h-6 w-6 items-center justify-center rounded-[4px] bg-[#2C2C2C] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
                title="下一句 (→)"
              >
                <ChevronRight className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="flex items-center justify-between gap-1.5 rounded-[4px] border border-white/[0.06] bg-[#181818] px-2 py-1 text-[11px] text-secondary">
              <span>入点</span>
              <input
                aria-label="片段开始时间"
                type="number"
                step={0.05}
                min={0}
                disabled={editsDisabled}
                value={startDraft}
                onChange={(e) => setStartDraft(e.target.value)}
                onBlur={commitStartDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    commitStartDraft();
                    e.currentTarget.blur();
                  } else if (e.key === "Escape") {
                    setStartDraft(segment.start.toFixed(2));
                    e.currentTarget.blur();
                  }
                }}
                className="w-14 bg-transparent text-right font-mono tabular-nums text-primary outline-none"
              />
            </label>
            <label className="flex items-center justify-between gap-1.5 rounded-[4px] border border-white/[0.06] bg-[#181818] px-2 py-1 text-[11px] text-secondary">
              <span>出点</span>
              <input
                aria-label="片段结束时间"
                type="number"
                step={0.05}
                min={Number((segment.start + 0.2).toFixed(2))}
                disabled={editsDisabled}
                value={endDraft}
                onChange={(e) => setEndDraft(e.target.value)}
                onBlur={commitEndDraft}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    commitEndDraft();
                    e.currentTarget.blur();
                  } else if (e.key === "Escape") {
                    setEndDraft(segment.end.toFixed(2));
                    e.currentTarget.blur();
                  }
                }}
                className="w-14 bg-transparent text-right font-mono tabular-nums text-primary outline-none"
              />
            </label>
          </div>
        </div>

        {/* 主控区 1：配音台词 */}
        <div className="space-y-1.5 border-t border-dashed border-white/[0.08] pt-2.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-secondary">配音台词</span>
            <span
              className={`font-mono tabular-nums ${
                usedUnits > recommendedUnits ? "text-status-warning" : "text-muted"
              }`}
              title={`建议 ≤ ${recommendedUnits}${unit}`}
            >
              {usedUnits}/{recommendedUnits}
              {unit}
            </span>
          </div>
          <textarea
            rows={2}
            disabled={editsDisabled}
            value={segment.target_text}
            onChange={(e) =>
              updateSegmentLocal({
                id: segment.id,
                target_text: e.target.value,
              })
            }
            className="w-full rounded-[4px] border border-white/[0.07] bg-[#181818] px-2.5 py-1.5 text-[12px] leading-relaxed text-primary transition-colors focus:border-white/25 focus:outline-none"
          />
        </div>

        {/* 主控区 2：情绪 · 说话人 · 语速 · 原声 */}
        <div className="space-y-2 border-t border-dashed border-white/[0.08] pt-2.5">
          <div className="grid h-[26px] grid-cols-5 gap-0.5 rounded-[4px] bg-[#181818] p-0.5">
            {EMOTIONS.map((emo) => {
              const active = segment.emotion === emo.id;
              return (
                <button
                  key={emo.id}
                  type="button"
                  title={emo.label}
                  disabled={editsDisabled}
                  onClick={() => updateSegmentLocal({ id: segment.id, emotion: emo.id })}
                  className={`flex items-center justify-center gap-1 whitespace-nowrap rounded-[3px] text-[11px] transition-colors disabled:opacity-50 ${
                    active
                      ? "bg-[#333333] font-medium text-[#FFFFFF]"
                      : "text-secondary hover:text-primary"
                  }`}
                >
                  <span
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{
                      backgroundColor: emo.dotColor,
                      opacity: active ? 1 : 0.45,
                    }}
                  />
                  <span>{emo.short}</span>
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2">
            <select
              aria-label="说话人"
              value={segment.speaker}
              disabled={editsDisabled}
              onChange={(e) => updateSegmentLocal({ id: segment.id, speaker: e.target.value })}
              className="h-7 min-w-0 flex-1 truncate rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 text-[12px] text-primary transition-colors focus:border-white/25 focus:outline-none"
            >
              {speakers.map((spk) => (
                <option key={spk} value={spk}>
                  {script.characters[spk]?.display_name ?? spk}
                </option>
              ))}
            </select>

            <label
              className="flex shrink-0 cursor-pointer items-center gap-1 rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 py-1 text-[11px] text-secondary"
              title="保留原声（笑声/叹息不替换配音）"
            >
              <input
                type="checkbox"
                disabled={editsDisabled}
                checked={segment.keep_original}
                onChange={(e) =>
                  updateSegmentLocal({
                    id: segment.id,
                    keep_original: e.target.checked,
                  })
                }
                className="h-3 w-3 rounded-[2px] disabled:opacity-50"
              />
              <span>原声</span>
            </label>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-8 shrink-0 text-[11px] text-secondary">语速</span>
            <input
              aria-label="语速倍率"
              type="range"
              disabled={editsDisabled}
              min={0.8}
              max={1.4}
              step={0.05}
              value={segment.speed_factor}
              onChange={(e) =>
                updateSegmentLocal({
                  id: segment.id,
                  speed_factor: Number(e.target.value),
                })
              }
              className="min-w-0 flex-1 disabled:opacity-50"
            />
            <span
              title={segment.speed_factor > 1.15 ? "加速偏高可能影响听感" : undefined}
              className={`w-10 shrink-0 text-right font-mono tabular-nums text-[11px] ${
                segment.speed_factor > 1.15 ? "text-status-warning" : "text-primary"
              }`}
            >
              {segment.speed_factor.toFixed(2)}x
            </span>
          </div>
        </div>

        {/* 主控区 3：时长进度与单句试听条 */}
        <div className="space-y-2 border-t border-dashed border-white/[0.08] pt-2.5">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-muted">时长</span>
              <span
                className={`font-mono tabular-nums ${
                  isOverflow || isTooShort
                    ? "text-status-error"
                    : actualDuration != null
                      ? "text-primary"
                      : "text-status-warning"
                }`}
              >
                {actualDuration != null
                  ? `${actualDuration.toFixed(1)} / ${slotDuration.toFixed(1)}s`
                  : `— / ${slotDuration.toFixed(1)}s`}
              </span>
            </div>
            <div className="relative h-1 w-full overflow-hidden rounded-[2px] bg-[#181818]">
              <div
                className={`h-full transition-all ${
                  isOverflow || isTooShort ? "bg-status-error" : "bg-[#A8A8A8]"
                }`}
                style={{
                  width: `${Math.min(100, Math.round((fillRatio / 1.3) * 100))}%`,
                }}
              />
              <div
                title={segment.applied_speed_factor == null ? "115% 超时预警" : "100% 上限"}
                style={{ left: `${Math.round((limitRatio / 1.3) * 100)}%` }}
                className="absolute bottom-0 top-0 w-px bg-status-warning"
              />
            </div>
          </div>

          {auditionUrl && (
            <>
              <audio
                key={auditionUrl}
                ref={audioRef}
                src={auditionUrl}
                preload="metadata"
                className="hidden"
                onLoadedMetadata={(e) => {
                  const dur = e.currentTarget.duration;
                  if (Number.isFinite(dur) && dur > 0) setAuditionDuration(dur);
                }}
                onTimeUpdate={(e) => setAuditionTime(e.currentTarget.currentTime)}
                onEnded={() => {
                  setIsAuditionPlaying(false);
                  setAuditionTime(0);
                }}
              />
              <div className="flex h-7 items-center gap-2 rounded-[4px] bg-[#2A2A2A] px-2">
                <button
                  type="button"
                  onClick={() => void toggleAuditionPlay()}
                  title={isAuditionPlaying ? "暂停" : "试听"}
                  className="flex h-5 w-5 shrink-0 items-center justify-center rounded-[3px] text-primary transition-colors hover:bg-[#363636]"
                >
                  {isAuditionPlaying ? (
                    <Pause className="h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
                  ) : (
                    <Play className="ml-0.5 h-3.5 w-3.5 fill-current" strokeWidth={1.5} />
                  )}
                </button>
                <div
                  role="slider"
                  aria-label="试听进度"
                  aria-valuemin={0}
                  aria-valuemax={Math.round(effectiveDuration)}
                  aria-valuenow={Math.round(auditionTime)}
                  onClick={handleAuditionSeek}
                  className="flex h-4 flex-1 cursor-pointer items-center"
                >
                  <div className="h-[3px] w-full overflow-hidden rounded-[2px] bg-[#3E3E3E]">
                    <div
                      className="h-full bg-[#B5B5B5] transition-[width] duration-75"
                      style={{ width: `${auditionProgress}%` }}
                    />
                  </div>
                </div>
                <span className="shrink-0 font-mono tabular-nums text-[10px] text-secondary">
                  {formatMiniTime(auditionTime)}/{formatMiniTime(effectiveDuration)}
                </span>
              </div>
            </>
          )}
        </div>

        {/* 折叠次要区：原文与校对 */}
        <details className="border-t border-dashed border-white/[0.08] pt-2.5 text-[11px]">
          <summary className="flex cursor-pointer select-none items-center justify-between text-secondary hover:text-primary">
            <span>原文与校对</span>
            {hasPendingReview && <span title="待确认" className="h-1.5 w-1.5 rounded-full bg-status-warning" />}
          </summary>
          <div className="mt-2 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-[10px] text-muted">
                {segment.alignment_granularity === "word"
                  ? "词级"
                  : segment.alignment_granularity === "segment"
                    ? "句段级"
                    : "未对齐"}
              </span>
              {(script.media.vocals_wav || script.media.source_video) && (
                <>
                  <audio
                    ref={sourceAudioRef}
                    src={script.media.vocals_wav || script.media.source_video || undefined}
                    preload="none"
                    onPause={() => setSourcePlaying(false)}
                    onEnded={() => setSourcePlaying(false)}
                    onTimeUpdate={(e) => {
                      if (e.currentTarget.currentTime >= segment.end) {
                        e.currentTarget.pause();
                        setSourcePlaying(false);
                      }
                    }}
                  />
                  <button
                    type="button"
                    title="试听原声"
                    onClick={async () => {
                      const audio = sourceAudioRef.current;
                      if (!audio) return;
                      if (!audio.paused) {
                        audio.pause();
                        setSourcePlaying(false);
                        return;
                      }
                      try {
                        audio.currentTime = segment.start;
                        await audio.play();
                        setSourcePlaying(true);
                      } catch {
                        setSynthNote("原声试听失败");
                      }
                    }}
                    className="flex h-6 items-center gap-1 rounded-[3px] bg-[#2A2A2A] px-2 text-secondary hover:text-primary"
                  >
                    {sourcePlaying ? (
                      <Pause className="h-3 w-3" strokeWidth={1.6} />
                    ) : (
                      <Volume2 className="h-3 w-3" strokeWidth={1.6} />
                    )}
                    <span>原声</span>
                  </button>
                </>
              )}
            </div>

            <textarea
              id="source-transcript"
              rows={2}
              disabled={editsDisabled}
              className="w-full rounded-[4px] border border-white/[0.07] bg-[#181818] px-2 py-1.5 text-[12px] text-primary"
              value={segment.orig_text}
              onChange={(e) => updateSegmentLocal({ id: segment.id, orig_text: e.target.value })}
            />

            {hasPendingReview && (
              <div className="flex flex-wrap gap-1.5">
                {segment.segment_timing_needs_review !== false && (
                  <button
                    type="button"
                    className="rounded border border-white/[0.08] bg-[#282828] px-2 py-0.5 text-[10px] text-status-warning hover:text-primary disabled:opacity-40"
                    disabled={editsDisabled || !segment.orig_text.trim()}
                    onClick={() =>
                      updateSegmentLocal({ id: segment.id, segment_timing_needs_review: false })
                    }
                  >
                    确认边界
                  </button>
                )}
                {segment.alignment_needs_review !== false && Boolean(segment.words?.length) && (
                  <button
                    type="button"
                    className="rounded border border-white/[0.08] bg-[#282828] px-2 py-0.5 text-[10px] text-status-warning hover:text-primary disabled:opacity-40"
                    disabled={editsDisabled || !segment.orig_text.trim()}
                    onClick={() =>
                      updateSegmentLocal({ id: segment.id, alignment_needs_review: false })
                    }
                  >
                    确认词级
                  </button>
                )}
                {segment.speaker_needs_review !== false && script.characters[segment.speaker] && (
                  <button
                    type="button"
                    className="rounded border border-white/[0.08] bg-[#282828] px-2 py-0.5 text-[10px] text-status-warning hover:text-primary disabled:opacity-40"
                    disabled={editsDisabled || segment.speaker === "UNKNOWN"}
                    onClick={() => updateSegmentLocal({ id: segment.id, speaker: segment.speaker })}
                  >
                    确认角色
                  </button>
                )}
              </div>
            )}

            {!!segment.words?.length && (
              <details className="text-[10px] text-muted">
                <summary className="cursor-pointer select-none">词级 ({segment.words.length})</summary>
                <ul className="mt-1 max-h-24 space-y-0.5 overflow-y-auto">
                  {segment.words.map((word, index) => (
                    <li key={`${segment.id}-${index}-${word.word}`}>
                      {word.word} ·{" "}
                      {word.start == null || word.end == null
                        ? "—"
                        : `${word.start.toFixed(2)}–${word.end.toFixed(2)}s`}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </details>

        <SourceEvidencePanel key={`${script.project_id}-${segment.id}`} script={script} segment={segment} />
      </div>

      {/* 底部操作栏 */}
      <div className="shrink-0 space-y-1.5 border-t border-white/[0.06] bg-[#222222] py-2 pl-3.5 pr-4">
        {synthNote && (
          <p className="truncate rounded-[4px] bg-white/[0.05] px-2 py-1 text-[11px] text-secondary">
            {synthNote}
          </p>
        )}
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={editsDisabled}
            onClick={() => void saveProjectScript()}
            title="保存 (Ctrl+S)"
            className="flex h-7 items-center justify-center gap-1.5 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-3 text-[12px] font-medium text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
            <span>保存</span>
          </button>
          <button
            type="button"
            onClick={() => void handleSingleSynthesize()}
            disabled={editsDisabled || segment.keep_original}
            className="flex h-7 flex-1 items-center justify-center gap-1.5 rounded-[4px] bg-[#3A6EA5] px-2.5 text-[12px] font-medium text-[#FFFFFF] transition-colors hover:bg-[#467DB8] disabled:opacity-50"
          >
            <Mic className="h-3.5 w-3.5" strokeWidth={1.5} />
            <span>{isSynthesizing ? "合成中" : "重配"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
