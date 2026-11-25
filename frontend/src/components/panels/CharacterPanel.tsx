import { Sparkles, Trash2, Upload, Wand2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  autoMatchEmotionRefs,
  createVoicePreset,
  deleteVoicePreset,
  fetchFinetuneStatus,
  fetchProjectScript,
  fetchVoicePresets,
  previewAutomaticReference,
  startCharacterFinetune,
  uploadReference,
} from "../../services/api";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";
import type {
  CharacterConfig,
  EmotionRef,
  EmotionType,
  FinetuneJobStatus,
  ProjectScript,
  VoicePreset,
} from "../../types/script";

const emotions: Record<EmotionType, string> = {
  neutral: "平淡",
  angry: "愤怒",
  happy: "喜悦",
  sad: "悲伤",
  whisper: "低语",
};

const FINETUNE_STAGE_LABELS: Record<string, string> = {
  idle: "空闲",
  preparing: "切分干声",
  step_1a_text: "文本特征",
  step_1b_ssl_sv: "声纹与SSL",
  step_1c_semantic: "语义编码",
  step_2_sovits: "微调 SoVITS",
  step_3_gpt: "微调 GPT",
  registering_preset: "存为预设",
  completed: "已完成",
  error: "失败",
};

const field = "min-w-0 w-full rounded border border-white/10 bg-[#181818] px-2 py-1 text-[11px] text-primary";
const button =
  "rounded bg-[#333333] px-2 py-1 text-[11px] text-primary hover:bg-[#404040] disabled:opacity-40";

function referenceUrl(wav: string): string | undefined {
  if (wav.startsWith("/assets/") || wav.startsWith("/workspace/")) return wav;
  if (wav.startsWith("assets/") || wav.startsWith("workspace/")) return `/${wav}`;
  if (wav.startsWith("refs/")) return `/assets/${wav}`;
}

function ReferenceEditor({
  script,
  speaker,
  emotion,
  character,
  busy,
}: {
  script: ProjectScript;
  speaker: string;
  emotion: EmotionType;
  character: CharacterConfig;
  busy: boolean;
}) {
  const [draft, setDraft] = useState<EmotionRef>(
    character.emotion_refs[emotion] ?? { wav: "", text: "", lang: script.source_lang ?? "" },
  );
  const [inFlight, setInFlight] = useState(false);
  const [message, setMessage] = useState("");
  const updateCharacter = useProjectStore((s) => s.updateCharacter);
  const audioUrl = referenceUrl(draft.wav);

  const boundRef = character.emotion_refs[emotion];
  const boundWav = boundRef?.wav ?? "";
  const boundText = boundRef?.text ?? "";
  const boundLang = boundRef?.lang ?? script.source_lang ?? "";
  useEffect(() => {
    setDraft({ wav: boundWav, text: boundText, lang: boundLang });
  }, [boundWav, boundText, boundLang]);

  async function loadAudio(file?: File) {
    setInFlight(true);
    setMessage("");
    try {
      if (file) {
        const { wav } = await uploadReference(script.project_id, file);
        setDraft((old) => ({ ...old, wav, text: "" }));
        setMessage("已上传，请补全逐字稿");
        return;
      }
      const state = useProjectStore.getState();
      if (state.isDirty) await state.saveProjectScript();
      if (useProjectStore.getState().script?.project_id !== script.project_id) return;
      const saved = useProjectStore.getState();
      if (saved.error || saved.isDirty || saved.backendStatus !== "online") {
        throw new Error(saved.error ?? "请先保存");
      }
      const { wav, text, lang, segment_ids } = await previewAutomaticReference(
        script.project_id,
        speaker,
      );
      setDraft({ wav, text, lang });
      setMessage(`已取自 #${segment_ids.join(",#")}`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "加载失败");
    } finally {
      setInFlight(false);
    }
  }

  const refs = character.emotion_refs;
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1">
        <label
          title="上传 3–10 秒参考音"
          className={`flex items-center gap-1 ${button} ${busy || inFlight ? "pointer-events-none opacity-40" : "cursor-pointer"}`}
        >
          <Upload className="h-3 w-3" strokeWidth={1.6} />
          <span>上传</span>
          <input
            className="sr-only"
            type="file"
            accept="audio/*,.wav,.flac,.m4a,.mp3,.opus"
            disabled={busy || inFlight}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void loadAudio(file);
              e.target.value = "";
            }}
          />
        </label>
        <button
          className={button}
          type="button"
          disabled={busy || inFlight}
          onClick={() => void loadAudio()}
        >
          提取参考
        </button>
      </div>
      {audioUrl && (
        <audio
          className="h-8 w-full"
          src={audioUrl}
          controls
          preload="none"
          onError={() => setMessage("音频不可播放")}
        />
      )}
      <textarea
        aria-label="参考音逐字稿"
        placeholder="参考音逐字稿"
        className={field}
        rows={2}
        value={draft.text}
        onChange={(e) => setDraft({ ...draft, text: e.target.value })}
      />
      <div className="flex items-center gap-1.5">
        <input
          aria-label="参考音语种"
          className={`${field} w-20`}
          placeholder="语种 ja/zh"
          value={draft.lang}
          onChange={(e) => setDraft({ ...draft, lang: e.target.value })}
        />
        <button
          className={button}
          type="button"
          disabled={inFlight || !draft.wav || !draft.text.trim() || !draft.lang.trim()}
          onClick={() => {
            updateCharacter(speaker, {
              emotion_refs: {
                ...refs,
                [emotion]: {
                  ...draft,
                  text: draft.text.trim(),
                  lang: draft.lang.trim().toLowerCase(),
                },
              },
            });
            setMessage("已应用");
          }}
        >
          应用
        </button>
        {refs[emotion] && (
          <button
            className={button}
            type="button"
            disabled={inFlight}
            onClick={() => {
              const { [emotion]: _removed, ...remaining } = refs;
              updateCharacter(speaker, { emotion_refs: remaining });
              setDraft({ wav: "", text: "", lang: script.source_lang ?? "" });
              setMessage("已移除");
            }}
          >
            移除
          </button>
        )}
      </div>
      {(inFlight || message) && (
        <p role="status" className="text-[10px]">
          {inFlight ? "处理中…" : message}
        </p>
      )}
    </div>
  );
}

function VoicePresetLibrary({
  speaker,
  character,
  busy,
  refreshKey = 0,
}: {
  speaker: string;
  character: CharacterConfig;
  busy: boolean;
  refreshKey?: number;
}) {
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const script = useProjectStore((s) => s.script);
  const isDirty = useProjectStore((s) => s.isDirty);
  const applyVoicePresetToCharacter = useProjectStore((s) => s.applyVoicePresetToCharacter);
  const [presets, setPresets] = useState<VoicePreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState("");

  const loadPresets = useCallback(async () => {
    if (backendStatus !== "online") return;
    setLoading(true);
    try {
      const res = await fetchVoicePresets();
      setPresets(res.presets);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [backendStatus]);

  useEffect(() => {
    void loadPresets();
  }, [loadPresets, refreshKey]);

  const handleSavePreset = async () => {
    const cleanName = presetName.trim() || character.display_name || speaker;
    setLoading(true);
    setNote("");
    try {
      const created = await createVoicePreset({
        name: cleanName,
        speaking_style: character.speaking_style ?? "",
        gpt_weights: character.gpt_weights ?? null,
        sovits_weights: character.sovits_weights ?? null,
        emotion_refs: character.emotion_refs,
      });
      setPresets((prev) => [created, ...prev]);
      setPresetName("");
      setNote(`已存“${created.name}”`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "保存失败");
    } finally {
      setLoading(false);
    }
  };

  const handleDeletePreset = async (preset: VoicePreset) => {
    const prefix = `/assets/refs/presets/${preset.preset_id}/`;
    const referencedLocally = Object.values(script?.characters ?? {}).some((char) =>
      Object.values(char.emotion_refs).some((ref) => ref?.wav.startsWith(prefix)),
    );
    const confirmMsg = referencedLocally
      ? `删除预设“${preset.name}”？${isDirty ? "建议先保存当前项目。" : "已保存项目音频仍会保留。"}`
      : `删除预设“${preset.name}”？`;
    if (!window.confirm(confirmMsg)) return;
    setLoading(true);
    setNote("");
    try {
      await deleteVoicePreset(preset.preset_id);
      setPresets((prev) => prev.filter((item) => item.preset_id !== preset.preset_id));
    } catch (err) {
      setNote(err instanceof Error ? err.message : "删除失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <details className="space-y-2 border-t border-white/10 pt-2 text-[11px]">
      <summary className="cursor-pointer select-none text-primary">预设 ({presets.length})</summary>
      <div className="flex gap-1">
        <input
          className={field}
          aria-label="预设名称"
          placeholder={character.display_name || speaker}
          value={presetName}
          onChange={(e) => setPresetName(e.target.value)}
        />
        <button
          className={`${button} shrink-0`}
          type="button"
          disabled={busy || loading || backendStatus !== "online"}
          onClick={() => void handleSavePreset()}
        >
          存预设
        </button>
      </div>
      {note && <div className="text-[10px] text-secondary">{note}</div>}
      {presets.length === 0 ? (
        <div className="text-[10px] text-muted">{backendStatus === "online" ? "暂无预设" : "未连接"}</div>
      ) : (
        <ul className="max-h-36 space-y-1 overflow-y-auto">
          {presets.map((preset) => (
            <li
              key={preset.preset_id}
              className="flex items-center justify-between gap-1 rounded bg-[#181818] px-2 py-1"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-primary">{preset.name}</div>
                <div className="truncate font-mono text-[10px] text-muted">
                  {Object.keys(preset.emotion_refs).length} 参
                  {preset.gpt_weights || preset.sovits_weights ? " · 权重" : ""}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <button
                  type="button"
                  disabled={busy || loading}
                  onClick={() => {
                    applyVoicePresetToCharacter(speaker, preset);
                    setNote(`已套用“${preset.name}”`);
                  }}
                  className={button}
                >
                  套用
                </button>
                <button
                  type="button"
                  title="删除预设"
                  disabled={busy || loading}
                  onClick={() => void handleDeletePreset(preset)}
                  className="rounded bg-[#2C2C2C] p-1 text-status-error hover:bg-[#3A2828] disabled:opacity-40"
                >
                  <Trash2 className="h-3 w-3" strokeWidth={1.6} />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}

function SpeakerEditor({
  script,
  speaker,
  busy,
}: { script: ProjectScript; speaker: string; busy: boolean }) {
  const character = script.characters[speaker];
  const updateCharacter = useProjectStore((s) => s.updateCharacter);
  const setScript = useProjectStore((s) => s.setScript);
  const mergeCharacters = useProjectStore((s) => s.mergeCharacters);
  const batchUpdateSegments = useProjectStore((s) => s.batchUpdateSegmentsLocal);
  const confirmAllPendingReviews = useProjectStore((s) => s.confirmAllPendingReviews);
  const selectSegment = useProjectStore((s) => s.selectSegment);
  const setCurrentTime = useProjectStore((s) => s.setCurrentTime);

  const [emotion, setEmotion] = useState<EmotionType>("neutral");
  const [destination, setDestination] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [matchingRefs, setMatchingRefs] = useState(false);
  const [matchMsg, setMatchMsg] = useState("");
  const [allowUnreviewed, setAllowUnreviewed] = useState(false);
  const [presetName, setPresetName] = useState("");
  const [startingFinetune, setStartingFinetune] = useState(false);
  const [finetuneMsg, setFinetuneMsg] = useState("");
  const [finetuneStatus, setFinetuneStatus] = useState<FinetuneJobStatus | null>(null);
  const [presetRefreshKey, setPresetRefreshKey] = useState(0);
  const handledCompletionRef = useRef<string | null>(null);

  const segments = script.segments.filter((seg) => seg.speaker === speaker);
  const ids = segments.filter((seg) => selected.includes(seg.id)).map((seg) => seg.id);

  const clipStats = useMemo(() => {
    const clean = segments.filter((seg) => {
      const dur = seg.end - seg.start;
      if (seg.keep_original || !seg.orig_text.trim() || dur < 1.0) return false;
      return !script.segments.some(
        (other) => other.id !== seg.id && other.start < seg.end && other.end > seg.start,
      );
    });
    const confirmed = clean.filter((seg) => seg.speaker_needs_review === false);
    const totalSec = clean.reduce((sum, seg) => sum + (seg.end - seg.start), 0);
    return {
      cleanCount: clean.length,
      confirmedCount: confirmed.length,
      unreviewedCount: clean.length - confirmed.length,
      totalSeconds: totalSec,
      isFewShotCandidate: totalSec >= 6.0 && (totalSec < 20.0 || clean.length < 6),
      belowMinimum: clean.length < 2 || totalSec < 6.0,
    };
  }, [script.segments, segments]);

  const flushDirtyIfNeeded = async () => {
    const state = useProjectStore.getState();
    if (state.isDirty) await state.saveProjectScript();
    const saved = useProjectStore.getState();
    if (saved.script?.project_id !== script.project_id) throw new Error("项目已切换");
    if (saved.error || saved.isDirty || saved.backendStatus !== "online") {
      throw new Error(saved.error ?? "请先保存");
    }
  };

  useEffect(() => {
    let cancelled = false;
    const syncStatus = async () => {
      try {
        const status = await fetchFinetuneStatus(script.project_id, speaker);
        if (cancelled) return;
        setFinetuneStatus(status);
        if (
          status.stage === "completed" &&
          status.job_id &&
          handledCompletionRef.current !== status.job_id
        ) {
          handledCompletionRef.current = status.job_id;
          const latest = await fetchProjectScript(script.project_id);
          if (!cancelled) {
            setScript(latest);
            setPresetRefreshKey((prev) => prev + 1);
            setFinetuneMsg("微调完成");
          }
        }
      } catch {
        // Ignore transient polling failures
      }
    };
    void syncStatus();
    if (!finetuneStatus?.active) {
      return () => {
        cancelled = true;
      };
    }
    const timer = window.setInterval(() => void syncStatus(), 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [finetuneStatus?.active, script.project_id, setScript, speaker]);

  const handleAutoMatchEmotionRefs = async () => {
    setMatchingRefs(true);
    setMatchMsg("");
    try {
      await flushDirtyIfNeeded();
      const res = await autoMatchEmotionRefs(script.project_id, speaker, false);
      setScript(res.script);
      if (res.populated_emotions.length > 0) {
        const labels = res.populated_emotions.map((key) => emotions[key as EmotionType] ?? key);
        setMatchMsg(`已配 ${labels.join("、")}`);
      } else {
        setMatchMsg("无新增匹配");
      }
    } catch (err) {
      setMatchMsg(err instanceof Error ? err.message : "匹配失败");
    } finally {
      setMatchingRefs(false);
    }
  };

  const handleStartFinetune = async () => {
    setStartingFinetune(true);
    setFinetuneMsg("");
    try {
      await flushDirtyIfNeeded();
      const status = await startCharacterFinetune(script.project_id, speaker, {
        preset_name: presetName.trim() || null,
        allow_unreviewed: allowUnreviewed,
        bind_to_character: true,
        save_as_preset: true,
      });
      handledCompletionRef.current = null;
      setFinetuneStatus(status);
    } catch (err) {
      setFinetuneMsg(err instanceof Error ? err.message : "启动失败");
    } finally {
      setStartingFinetune(false);
    }
  };

  const configuredRefCount = Object.values(character.emotion_refs).filter(
    (ref) => ref?.wav && ref.text?.trim() && ref.lang?.trim(),
  ).length;
  const pendingReviewCount = segments.filter(
    (seg) =>
      seg.segment_timing_needs_review !== false ||
      seg.alignment_needs_review !== false ||
      (seg.speaker_needs_review !== false && seg.speaker !== "UNKNOWN"),
  ).length;

  return (
    <div className="space-y-2 text-[11px]">
      <div className="grid grid-cols-2 gap-1.5">
        <input
          aria-label="角色名"
          placeholder="角色名"
          className={field}
          value={character.display_name}
          onChange={(e) => updateCharacter(speaker, { display_name: e.target.value })}
        />
        <input
          aria-label="说话语气"
          placeholder="语气描述"
          className={field}
          value={character.speaking_style ?? ""}
          onChange={(e) => updateCharacter(speaker, { speaking_style: e.target.value })}
        />
      </div>

      <div className="flex items-center justify-between gap-1.5">
        <button
          className={`flex flex-1 items-center justify-center gap-1 ${button}`}
          type="button"
          title="将该角色下所有待校对句子的时间、原文与说话人一次性标为已确认"
          disabled={busy || speaker === "UNKNOWN" || pendingReviewCount === 0}
          onClick={() => confirmAllPendingReviews(speaker)}
        >
          <span>
            {pendingReviewCount > 0 ? `一键确认本角色 (${pendingReviewCount})` : "本角色已确认"}
          </span>
        </button>
      </div>

      <details className="space-y-2 border-t border-white/10 pt-2" open>
        <summary className="flex cursor-pointer select-none items-center justify-between text-primary">
          <span>参考音</span>
          <span className="font-mono text-[10px] text-muted">{configuredRefCount}/5</span>
        </summary>
        <div className="flex items-center gap-1.5">
          <button
            className={`flex items-center gap-1 ${button}`}
            type="button"
            title="声学分析并自动匹配五种情绪参考音"
            disabled={busy || matchingRefs || speaker === "UNKNOWN"}
            onClick={() => void handleAutoMatchEmotionRefs()}
          >
            <Wand2 className="h-3 w-3" strokeWidth={1.6} />
            <span>{matchingRefs ? "匹配中…" : "自动匹配"}</span>
          </button>
          {matchMsg && <span className="truncate text-[10px] text-secondary">{matchMsg}</span>}
        </div>
        <select
          className={field}
          aria-label="参考音情绪"
          value={emotion}
          onChange={(e) => setEmotion(e.target.value as EmotionType)}
        >
          {Object.entries(emotions).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
              {character.emotion_refs[key as EmotionType] ? " · 已配置" : ""}
            </option>
          ))}
        </select>
        <ReferenceEditor
          key={emotion}
          script={script}
          speaker={speaker}
          emotion={emotion}
          character={character}
          busy={busy}
        />
        <details className="pt-1 text-[10px] text-muted">
          <summary className="cursor-pointer select-none hover:text-primary">自定义权重</summary>
          <div className="mt-1.5 space-y-1.5">
            <input
              aria-label="GPT 权重路径"
              placeholder="GPT .ckpt"
              className={field}
              value={character.gpt_weights ?? ""}
              onChange={(e) => updateCharacter(speaker, { gpt_weights: e.target.value || null })}
            />
            <input
              aria-label="SoVITS 权重路径"
              placeholder="SoVITS .pth"
              className={field}
              value={character.sovits_weights ?? ""}
              onChange={(e) => updateCharacter(speaker, { sovits_weights: e.target.value || null })}
            />
          </div>
        </details>
      </details>

      <details
        className="space-y-2 border-t border-white/10 pt-2"
        open={Boolean(finetuneStatus && finetuneStatus.stage !== "idle")}
      >
        <summary className="flex cursor-pointer select-none items-center justify-between text-primary">
          <span className="flex items-center gap-1">
            <Sparkles className="h-3 w-3 text-secondary" strokeWidth={1.6} />
            <span>角色微调</span>
          </span>
          <span className="font-mono text-[10px] text-muted">
            {clipStats.cleanCount}句/{clipStats.totalSeconds.toFixed(0)}s
          </span>
        </summary>
        <label
          className="flex items-center gap-1.5 text-[11px] text-secondary"
          title="未校对片段可能混入串角声音"
        >
          <input
            type="checkbox"
            checked={allowUnreviewed}
            disabled={busy || startingFinetune || Boolean(finetuneStatus?.active)}
            onChange={(e) => setAllowUnreviewed(e.target.checked)}
          />
          <span>
            含未校对片段 ({clipStats.confirmedCount}/{clipStats.cleanCount})
          </span>
        </label>
        <div className="flex gap-1">
          <input
            className={field}
            aria-label="微调预设名称"
            placeholder={character.display_name || speaker}
            value={presetName}
            disabled={busy || startingFinetune || Boolean(finetuneStatus?.active)}
            onChange={(e) => setPresetName(e.target.value)}
          />
          <button
            className="shrink-0 rounded bg-[#3A6EA5] px-2.5 py-1 text-white hover:bg-[#467CB5] disabled:opacity-40"
            type="button"
            disabled={
              busy ||
              startingFinetune ||
              speaker === "UNKNOWN" ||
              Boolean(finetuneStatus?.active)
            }
            onClick={() => void handleStartFinetune()}
          >
            {finetuneStatus?.active ? "训练中" : startingFinetune ? "准备中" : "开始微调"}
          </button>
        </div>
        {clipStats.isFewShotCandidate && (
          <div className="text-[10px] text-secondary">
            少样本防过拟合模式：自动冻结 GPT 仅微调 SoVITS 音色
          </div>
        )}
        {clipStats.belowMinimum && (
          <div className="text-[10px] text-muted">
            微调至少需 2 句且累计 ≥ 6s 无重叠干声
          </div>
        )}
        {finetuneMsg && <div className="text-[10px] text-secondary">{finetuneMsg}</div>}
        {finetuneStatus && finetuneStatus.stage !== "idle" && (
          <div className="space-y-1 rounded bg-[#181818] p-2 text-[10px]">
            <div className="flex items-center justify-between">
              <span className="font-medium text-primary">
                {FINETUNE_STAGE_LABELS[finetuneStatus.stage] ?? finetuneStatus.stage}
              </span>
              <span className="font-mono tabular-nums text-muted">
                {finetuneStatus.progress_pct}%
              </span>
            </div>
            <div className="h-1 w-full overflow-hidden rounded bg-[#2A2A2A]">
              <div
                className="h-full bg-[#3A6EA5] transition-all"
                style={{ width: `${Math.min(100, Math.max(0, finetuneStatus.progress_pct))}%` }}
              />
            </div>
            {finetuneStatus.error && (
              <div className="break-all text-status-error">{finetuneStatus.error}</div>
            )}
            {finetuneStatus.logs.length > 0 && (
              <div className="max-h-16 overflow-y-auto font-mono text-muted">
                {finetuneStatus.logs.slice(-3).map((line, idx) => (
                  <div key={idx} className="truncate">
                    {line}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </details>

      <VoicePresetLibrary
        speaker={speaker}
        character={character}
        busy={busy}
        refreshKey={presetRefreshKey}
      />

      <details className="space-y-2 border-t border-white/10 pt-2">
        <summary className="cursor-pointer select-none text-primary">
          说话人校对 ({segments.length})
        </summary>
        {segments.length > 0 && (
          <>
            <label className="flex items-center gap-2 text-secondary">
              <input
                type="checkbox"
                checked={ids.length === segments.length}
                onChange={(e) => setSelected(e.target.checked ? segments.map((seg) => seg.id) : [])}
              />
              全选
            </label>
            <ul className="max-h-36 space-y-1 overflow-y-auto">
              {segments.map((seg) => (
                <li key={seg.id} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    aria-label={`选择片段 ${seg.id}`}
                    checked={selected.includes(seg.id)}
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? [...selected, seg.id]
                          : selected.filter((id) => id !== seg.id),
                      )
                    }
                  />
                  <button
                    type="button"
                    className="min-w-0 flex-1 truncate text-left hover:text-primary"
                    onClick={() => {
                      selectSegment(seg.id);
                      setCurrentTime(seg.start);
                    }}
                  >
                    #{seg.id} {seg.speaker_needs_review !== false ? "· " : ""}
                    {seg.orig_text}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
        <select
          aria-label="目标角色"
          className={field}
          value={destination || speaker}
          onChange={(e) => setDestination(e.target.value)}
        >
          {Object.entries(script.characters).map(([id, item]) => (
            <option key={id} value={id}>
              {item.display_name || id}
            </option>
          ))}
        </select>
        <div className="flex flex-wrap gap-1">
          <button
            className={button}
            type="button"
            disabled={!ids.length}
            onClick={() => {
              batchUpdateSegments(ids.map((id) => ({ id, speaker: destination || speaker })));
              setSelected([]);
            }}
          >
            {!destination || destination === speaker ? "确认" : "改派"}
          </button>
          <button
            className={button}
            type="button"
            disabled={!destination || destination === speaker}
            onClick={() => mergeCharacters(speaker, destination)}
          >
            合并角色
          </button>
        </div>
      </details>
    </div>
  );
}

export function CharacterPanel({ script }: { script: ProjectScript }) {
  const [speakerKey, setSpeakerKey] = useState("");
  const [name, setName] = useState("");
  const addCharacter = useProjectStore((s) => s.addCharacter);
  const offline = useProjectStore((s) => s.backendStatus !== "online");
  const isLoading = useProjectStore((s) => s.isLoading || s.isSynthesizing);
  const pipelineRunning = useWorkflowStore((s) => s.isPipelineRunning);
  const speaker = script.characters[speakerKey] ? speakerKey : Object.keys(script.characters)[0];

  return (
    <fieldset
      className="min-w-0 space-y-2.5 p-3 text-secondary"
      disabled={isLoading || pipelineRunning}
    >
      <form
        className="flex gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          const key = addCharacter(name);
          if (key) {
            setSpeakerKey(key);
            setName("");
          }
        }}
      >
        <input
          className={field}
          aria-label="新角色名称"
          placeholder="新建角色"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button className={`${button} shrink-0`} type="submit" disabled={!name.trim()}>
          +
        </button>
      </form>
      {speaker && (
        <>
          <select
            className={field}
            aria-label="选择要编辑的角色"
            value={speaker}
            onChange={(e) => setSpeakerKey(e.target.value)}
          >
            {Object.entries(script.characters).map(([id, character]) => (
              <option key={id} value={id}>
                {character.display_name || id}
              </option>
            ))}
          </select>
          <SpeakerEditor
            key={speaker}
            script={script}
            speaker={speaker}
            busy={offline || isLoading || pipelineRunning}
          />
        </>
      )}
    </fieldset>
  );
}
