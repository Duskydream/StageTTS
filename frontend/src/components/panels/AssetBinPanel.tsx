import { Check, Volume2, X } from "lucide-react";
import type React from "react";
import { useState } from "react";
import { getSubtitlesUrl } from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type { StemVolumes, SubtitleLayout } from "../../types/script";
import { isSegmentOverflow, isSegmentTooShort } from "../../utils/synthesisIssues";
import { CharacterPanel } from "./CharacterPanel";
import { TranslationSettings } from "./TranslationSettings";

type DrawerSegmentFilter = "all" | "review" | "pending" | "issue";

/**
 * 240px Premiere 项目抽屉（折叠分组 + 图标/计数筛选）。
 */
export const AssetBinPanel: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isDirty = useProjectStore((s) => s.isDirty);
  const isLoading = useProjectStore((s) => s.isLoading);
  const isSynthesizing = useProjectStore((s) => s.isSynthesizing);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);
  const saveProjectScript = useProjectStore((s) => s.saveProjectScript);
  const updateStemVolume = useProjectStore((s) => s.updateStemVolume);
  const updateAutoDucking = useProjectStore((s) => s.updateAutoDucking);
  const confirmAllPendingReviews = useProjectStore((s) => s.confirmAllPendingReviews);

  const leftDrawer = useWorkflowStore((s) => s.leftDrawer);
  const isPipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const toggleLeftDrawer = useWorkflowStore((s) => s.toggleLeftDrawer);
  const [segmentFilter, setSegmentFilter] = useState<DrawerSegmentFilter>("all");
  const [subNote, setSubNote] = useState<string | null>(null);

  if (!script || leftDrawer === "none") {
    return null;
  }

  const editsDisabled = isLoading || isSynthesizing || isPipelineRunning;

  const handleDownloadSubtitles = async (layout: SubtitleLayout) => {
    if (backendStatus !== "online") {
      setSubNote("离线不可导出");
      return;
    }
    setSubNote(null);
    if (isDirty) {
      await saveProjectScript();
      const afterSave = useProjectStore.getState();
      if (afterSave.error || afterSave.isDirty) {
        setSubNote(afterSave.error ?? "请先保存");
        return;
      }
    }
    const link = document.createElement("a");
    link.href = getSubtitlesUrl(script.project_id, layout);
    link.download = `${script.project_id}_${layout}.ass`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const reviewPendingCount = script.segments.filter(
    (seg) => seg.alignment_needs_review || seg.speaker_needs_review,
  ).length;
  const pendingSynthCount = script.segments.filter(
    (seg) => !seg.keep_original && seg.status !== "ready",
  ).length;
  const issueCount = script.segments.filter(
    (seg) => seg.status === "error" || isSegmentOverflow(seg) || isSegmentTooShort(seg),
  ).length;
  const filteredSegments = script.segments.filter((seg) => {
    if (segmentFilter === "all") return true;
    if (segmentFilter === "review") {
      return Boolean(seg.alignment_needs_review || seg.speaker_needs_review);
    }
    if (segmentFilter === "pending") {
      return !seg.keep_original && seg.status !== "ready";
    }
    return seg.status === "error" || isSegmentOverflow(seg) || isSegmentTooShort(seg);
  });

  const stems = script.media.cinematic_stems;
  const stemVolumes = script.media.stem_volumes ?? { singing: 1, music: 1, effects: 1 };
  const autoDucking = script.media.auto_ducking ?? true;
  const stemControls: Array<{ key: keyof StemVolumes; label: string }> = [
    { key: "singing", label: "歌声" },
    { key: "music", label: "音乐" },
    { key: "effects", label: "音效" },
  ];
  const mediaRows = [
    { label: "原片", ready: Boolean(script.media.source_video) },
    ...(stems
      ? [
          { label: "对白", ready: Boolean(stems.dialogue), url: stems.dialogue },
          { label: "歌声", ready: Boolean(stems.singing), url: stems.singing },
          { label: "音乐", ready: Boolean(stems.music), url: stems.music },
          { label: "音效", ready: Boolean(stems.effects), url: stems.effects },
          { label: "背景", ready: Boolean(script.media.bgm_wav), url: script.media.bgm_wav ?? undefined },
        ]
      : [
          { label: "人声", ready: Boolean(script.media.vocals_wav) },
          { label: "背景", ready: Boolean(script.media.bgm_wav) },
        ]),
    { label: "成片", ready: Boolean(script.media.output_video) },
  ];

  return (
    <div className="flex h-full w-60 shrink-0 flex-col bg-[#222222] text-[12px]">
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-white/[0.06] px-3.5">
        <span className="text-[12px] font-medium text-primary">
          {leftDrawer === "characters" ? "角色" : "素材"}
        </span>
        <button
          type="button"
          title="收起"
          aria-label="收起"
          onClick={() => toggleLeftDrawer(leftDrawer)}
          className="flex h-6 w-6 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#363636] hover:text-primary"
        >
          <X className="h-3.5 w-3.5" strokeWidth={1.5} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {leftDrawer === "characters" ? (
          <CharacterPanel key={script.project_id} script={script} />
        ) : (
          <div>
            <TranslationSettings key={script.project_id} script={script} />

            <details className="border-b border-dashed border-white/[0.08] px-3.5 py-2.5 text-[11px]">
              <summary className="cursor-pointer select-none text-primary">媒体轨</summary>
              <div className="mt-2 divide-y divide-white/[0.05]">
                {mediaRows.map((row) => (
                  <div key={row.label} className="flex h-7 items-center justify-between">
                    <span className="text-secondary">{row.label}</span>
                    <span className="flex items-center gap-1.5">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${row.ready ? "bg-status-ready" : "bg-[#555555]"}`}
                      />
                      {"url" in row && row.url && (
                        <a
                          href={row.url}
                          target="_blank"
                          rel="noreferrer"
                          title={`试听${row.label}`}
                          aria-label={`试听${row.label}`}
                          className="text-secondary hover:text-primary"
                        >
                          <Volume2 className="h-3 w-3" strokeWidth={1.5} />
                        </a>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </details>

            <details className="border-b border-dashed border-white/[0.08] px-3.5 py-2.5 text-[11px]">
              <summary className="cursor-pointer select-none text-primary">混音与字幕</summary>
              <div className="mt-2 space-y-2">
                {stems && (
                  <div className="space-y-1.5">
                    {stemControls.map((item) => {
                      const value = stemVolumes[item.key] ?? 1;
                      return (
                        <div key={item.key} className="flex items-center gap-2">
                          <span className="w-7 shrink-0 text-secondary">{item.label}</span>
                          <input
                            aria-label={`${item.label}轨音量`}
                            type="range"
                            min={0}
                            max={2}
                            step={0.05}
                            value={value}
                            onChange={(e) => updateStemVolume(item.key, Number(e.target.value))}
                            className="min-w-0 flex-1"
                          />
                          <span className="w-8 shrink-0 text-right font-mono tabular-nums text-[10px] text-secondary">
                            {Math.round(value * 100)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
                <label className="flex cursor-pointer items-center justify-between gap-2 text-primary">
                  <span className="text-secondary">人声闪避</span>
                  <input
                    type="checkbox"
                    checked={autoDucking}
                    onChange={(e) => updateAutoDucking(e.target.checked)}
                    className="h-3.5 w-3.5"
                  />
                </label>

                <div className="flex items-center justify-between gap-2 pt-0.5">
                  <span className="text-secondary">ASS</span>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      disabled={editsDisabled || backendStatus !== "online"}
                      onClick={() => void handleDownloadSubtitles("target")}
                      className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                    >
                      单语
                    </button>
                    <button
                      type="button"
                      disabled={editsDisabled || backendStatus !== "online"}
                      onClick={() => void handleDownloadSubtitles("bilingual")}
                      className="rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-2 py-0.5 text-primary transition-colors hover:bg-[#363636] disabled:opacity-40"
                    >
                      双语
                    </button>
                  </div>
                </div>
                {subNote && <div className="text-status-warning">{subNote}</div>}
              </div>
            </details>

            <details className="p-3.5" open>
              <summary className="mb-2 flex cursor-pointer select-none items-center justify-between gap-2 text-[11px] font-medium text-primary">
                <span>台词 ({script.segments.length})</span>
                {reviewPendingCount > 0 && (
                  <button
                    type="button"
                    disabled={editsDisabled}
                    title={`一键确认全部 ${reviewPendingCount} 条待校对`}
                    onClick={(e) => {
                      e.preventDefault();
                      confirmAllPendingReviews();
                    }}
                    className="flex items-center gap-1 rounded-[3px] border border-white/[0.08] bg-[#2C2C2C] px-1.5 py-0.5 font-mono text-[10px] text-primary hover:bg-[#363636] disabled:opacity-40"
                  >
                    <Check className="h-3 w-3 text-status-ready" strokeWidth={1.6} />
                    <span>{reviewPendingCount}</span>
                  </button>
                )}
              </summary>

              <div className="mb-2 grid grid-cols-4 gap-0.5 rounded-[4px] bg-[#181818] p-0.5 text-[10px]">
                {(
                  [
                    { id: "all", label: "全部", count: script.segments.length },
                    { id: "review", label: "校对", count: reviewPendingCount },
                    { id: "pending", label: "待配", count: pendingSynthCount },
                    { id: "issue", label: "异常", count: issueCount },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setSegmentFilter(tab.id)}
                    className={`rounded-[3px] px-1 py-1 font-mono tabular-nums transition-colors ${
                      segmentFilter === tab.id
                        ? "bg-[#333333] text-[#FFFFFF]"
                        : "text-secondary hover:text-primary"
                    }`}
                  >
                    {tab.label}({tab.count})
                  </button>
                ))}
              </div>

              <div className="space-y-0.5">
                {filteredSegments.length === 0 ? (
                  <div className="py-3 text-center text-[11px] text-muted">无匹配</div>
                ) : (
                  filteredSegments.map((seg) => {
                    const needsReview = Boolean(
                      seg.alignment_needs_review || seg.speaker_needs_review,
                    );
                    const overflow = isSegmentOverflow(seg);
                    const tooShort = isSegmentTooShort(seg);
                    return (
                      <button
                        key={seg.id}
                        type="button"
                        onClick={() => {
                          selectSegment(seg.id);
                          setCurrentTime(seg.start);
                        }}
                        className="flex h-7 w-full items-center justify-between gap-1.5 rounded-[4px] px-2 text-left text-[11px] text-secondary transition-colors hover:bg-[#2C2C2C] hover:text-primary"
                      >
                        <span className="min-w-0 flex-1 truncate">
                          <span className="font-mono tabular-nums text-muted">#{seg.id}</span>{" "}
                          {seg.target_text}
                        </span>
                        <span className="flex shrink-0 items-center gap-1">
                          {needsReview && <span title="待校对" className="h-1.5 w-1.5 rounded-full bg-status-warning" />}
                          {(overflow || tooShort) && (
                            <span
                              title={overflow ? "超时" : "过短"}
                              className={`h-1.5 w-1.5 rounded-full ${overflow ? "bg-status-error" : "bg-status-warning"}`}
                            />
                          )}
                          <span className="font-mono tabular-nums text-[10px] text-muted">
                            {seg.start.toFixed(1)}s
                          </span>
                        </span>
                      </button>
                    );
                  })
                )}
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );
};
