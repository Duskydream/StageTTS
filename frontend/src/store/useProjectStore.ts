import { create } from "zustand";
import {
  ApiError,
  restoreProjectSnapshot as apiRestoreProjectSnapshot,
  saveProjectScript as apiSaveProjectScript,
  synthesizeSegments as apiSynthesizeSegments,
  checkHealth,
  fetchProjectScript,
  fetchProjects,
  translateProject,
  uploadProjectMedia,
} from "../services/api";
import type {
  CharacterConfig,
  ProjectMode,
  ProjectScript,
  Segment,
  SegmentPatch,
  StemVolumes,
  VoicePreset,
} from "../types/script";
import { synthesisIssues } from "../utils/synthesisIssues";

/**
 * 为静态音频/视频资源 URL 拼接防缓存时间戳（ARCHITECTURE.md 铁律 #8）。
 * 防止单句重合成覆盖本地文件后，浏览器仍播放缓存的旧音频。
 */
export function getCacheBustedUrl(
  rawUrl: string | null | undefined,
  versionToken?: number,
): string | null {
  if (!rawUrl) {
    return null;
  }
  if (!versionToken) {
    return rawUrl;
  }
  const separator = rawUrl.includes("?") ? "&" : "?";
  return `${rawUrl}${separator}t=${versionToken}`;
}

export type BackendStatus = "connecting" | "online" | "offline";

export interface TrackStates {
  videoMuted: boolean;
  dubMuted: boolean;
  dubSolo: boolean;
  vocalsMuted: boolean;
  bgmMuted: boolean;
}

interface ProjectState {
  script: ProjectScript | null;
  mode: ProjectMode;
  selectedSegmentId: number | null;
  currentTime: number;
  isPlaying: boolean;
  isLoading: boolean;
  isSynthesizing: boolean;
  isDirty: boolean;
  error: string | null;
  lastSyncMessage: string | null;
  backendStatus: BackendStatus;
  gpuBusy: boolean;
  projectList: string[];
  trackStates: TrackStates;
  timelineZoom: number;
  /** 记录每个片段最后一次重合成的时间戳，用于音频防缓存 */
  audioVersionMap: Record<number, number>;
  pastScripts: ProjectScript[];
  futureScripts: ProjectScript[];
  lastUndoMergeKey: { key: string; ts: number } | null;

  initWorkspace: () => Promise<void>;
  closeProject: () => void;
  importMediaFile: (file: File) => Promise<void>;
  setScript: (script: ProjectScript) => void;
  setMode: (mode: ProjectMode) => void;
  selectSegment: (segmentId: number | null) => void;
  stepSegment: (delta: -1 | 1) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setTimelineZoom: (zoom: number) => void;
  toggleTrackState: (key: keyof TrackStates) => void;
  bumpAudioVersion: (segmentIds: number[]) => void;
  commitHistoryCheckpoint: () => void;
  undo: () => boolean;
  redo: () => boolean;
  updateSegmentLocal: (patch: SegmentPatch, options?: { skipHistory?: boolean }) => void;
  batchUpdateSegmentsLocal: (patches: SegmentPatch[]) => number[];
  splitSegment: (segmentId: number, splitTime?: number) => number | null;
  mergeSegmentWithNext: (segmentId: number) => boolean;
  addSegment: (options?: { start?: number; end?: number; speaker?: string }) => number | null;
  deleteSegment: (segmentId: number) => boolean;
  confirmAllPendingReviews: (speakerKey?: string) => number;
  applyVoicePresetToCharacter: (speakerKey: string, preset: VoicePreset) => void;
  restoreSnapshot: (snapshotId: string) => Promise<boolean>;
  updateBgmVolume: (volume: number) => void;
  updateStemVolume: (stem: keyof StemVolumes, volume: number) => void;
  updateAutoDucking: (enabled: boolean) => void;
  updateCharacter: (speakerKey: string, patch: Partial<CharacterConfig>) => void;
  addCharacter: (name: string) => string | null;
  mergeCharacters: (source: string, destination: string) => void;
  updateTranslationSettings: (sourceLang: string | null, glossary: Record<string, string>) => void;
  retranslate: () => Promise<string>;
  loadProjectScript: (projectId: string) => Promise<void>;
  saveProjectScript: () => Promise<void>;
  synthesizeSegment: (segmentIds?: number[]) => Promise<{
    simulated: boolean;
    message: string;
    issues?: string[];
  }>;
}

const DEFAULT_TRACK_STATES: TrackStates = {
  videoMuted: false,
  dubMuted: false,
  dubSolo: false,
  vocalsMuted: true,
  bgmMuted: false,
};

function invalidateAudio(seg: Segment): Segment {
  return {
    ...seg,
    status: seg.keep_original ? "ready" : "pending",
    tts_wav: null,
    actual_duration: seg.keep_original ? seg.end - seg.start : null,
    applied_speed_factor: null,
  };
}

const MAX_HISTORY_STEPS = 50;
const UNDO_MERGE_WINDOW_MS = 800;

function pushPastScript(past: ProjectScript[], snapshot: ProjectScript): ProjectScript[] {
  const next = [...past, snapshot];
  return next.length > MAX_HISTORY_STEPS ? next.slice(next.length - MAX_HISTORY_STEPS) : next;
}

const CJK_BOUNDARY_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]/;

function joinSegmentText(left: string, right: string): string {
  const a = left.trim();
  const b = right.trim();
  if (!a) return b;
  if (!b) return a;
  if (CJK_BOUNDARY_RE.test(a[a.length - 1]) || CJK_BOUNDARY_RE.test(b[0])) {
    return `${a}${b}`;
  }
  return `${a} ${b}`;
}

function splitOrigTextByWords(
  origText: string,
  words: Segment["words"],
  splitWordIdx: number,
  ratio: number,
): [string, string] {
  if (words.length > 0 && splitWordIdx > 0 && splitWordIdx < words.length && origText) {
    const letters: Array<[number, string]> = [];
    for (let i = 0; i < origText.length; i += 1) {
      if (!/\s/.test(origText[i])) {
        letters.push([i, origText[i]]);
      }
    }
    const compact = letters.map(([, ch]) => ch).join("");
    let cursor = 0;
    const spans: Array<[number, number]> = [];
    let aligned = true;
    for (const w of words) {
      const token = w.word.replace(/\s+/g, "");
      if (!token || !compact.startsWith(token, cursor)) {
        aligned = false;
        break;
      }
      const startChar = letters[cursor][0];
      cursor += token.length;
      const endChar = letters[cursor - 1][0] + 1;
      spans.push([startChar, endChar]);
    }
    if (aligned && cursor === compact.length) {
      const leftEnd = spans[splitWordIdx - 1][1];
      const rightStart = spans[splitWordIdx][0];
      return [origText.slice(0, leftEnd).trim(), origText.slice(rightStart).trim()];
    }
  }
  return splitTextNearRatio(origText, ratio);
}

function splitTextNearRatio(text: string, ratio: number): [string, string] {
  const trimmed = text.trim();
  if (trimmed.length < 2) {
    return [trimmed, ""];
  }
  const clampedRatio = Math.max(0.15, Math.min(0.85, ratio));
  const rawIdx = Math.round(trimmed.length * clampedRatio);
  const spaceIdx = trimmed.indexOf(" ", Math.max(1, rawIdx - 4));
  const splitIdx =
    spaceIdx !== -1 && spaceIdx <= rawIdx + 6
      ? spaceIdx
      : Math.max(1, Math.min(trimmed.length - 1, rawIdx));
  return [trimmed.slice(0, splitIdx).trim(), trimmed.slice(splitIdx).trim()];
}

function applyPatchToSegments(
  script: ProjectScript,
  patches: SegmentPatch[],
): { nextScript: ProjectScript; markedPendingIds: number[] } {
  const resynthFields: Array<keyof SegmentPatch & keyof Segment> = [
    "start",
    "end",
    "speaker",
    "target_text",
    "target_lang",
    "emotion",
    "keep_original",
    "speed_factor",
  ];
  const patchMap = new Map<number, SegmentPatch>();
  for (const p of patches) {
    patchMap.set(p.id, p);
  }
  const markedPendingIds: number[] = [];
  let timingChanged = false;

  const nextSegments = script.segments.map((seg) => {
    const patch = patchMap.get(seg.id);
    if (!patch) {
      return seg;
    }
    const { id: _ignoreId, ...fields } = patch;
    const cleanFields = Object.fromEntries(
      Object.entries(fields).filter(([, value]) => value !== undefined),
    ) as Partial<Segment>;
    if (cleanFields.start !== undefined || cleanFields.end !== undefined) {
      const nextStart = Math.max(0, Number((cleanFields.start ?? seg.start).toFixed(2)));
      const nextEnd = Number((cleanFields.end ?? seg.end).toFixed(2));
      if (nextEnd <= nextStart + 0.05) {
        if (cleanFields.start !== undefined && cleanFields.end === undefined) {
          cleanFields.start = Math.max(0, Number((nextEnd - 0.2).toFixed(2)));
        } else if (cleanFields.end !== undefined && cleanFields.start === undefined) {
          cleanFields.end = Number((nextStart + 0.2).toFixed(2));
        } else {
          cleanFields.start = seg.start;
          cleanFields.end = seg.end;
        }
      } else {
        cleanFields.start = nextStart;
        cleanFields.end = nextEnd;
      }
      if (cleanFields.start !== seg.start || cleanFields.end !== seg.end) {
        timingChanged = true;
      }
    }
    if (cleanFields.speaker !== undefined) {
      if (!Object.prototype.hasOwnProperty.call(script.characters, cleanFields.speaker)) return seg;
      cleanFields.speaker_needs_review = cleanFields.speaker === "UNKNOWN"
        ? true
        : cleanFields.speaker_needs_review ?? false;
      if (cleanFields.speaker !== "UNKNOWN") cleanFields.speaker_fallback_reason = null;
    }
    if (cleanFields.orig_text !== undefined && cleanFields.orig_text !== seg.orig_text) {
      cleanFields.alignment_needs_review ??= true;
      cleanFields.segment_timing_needs_review = true;
    }
    if (cleanFields.speaker !== undefined && cleanFields.speaker !== seg.speaker) {
      cleanFields.segment_timing_needs_review = true;
    }
    const updated = { ...seg, ...cleanFields };
    const affectsAudio = resynthFields.some((key) => updated[key] !== seg[key]);
    if (affectsAudio && !updated.keep_original) {
      markedPendingIds.push(seg.id);
    }
    return affectsAudio ? invalidateAudio(updated) : updated;
  });

  const segments = timingChanged
    ? [...nextSegments].sort((a, b) => a.start - b.start || a.id - b.id)
    : nextSegments;

  return {
    nextScript: {
      ...script,
      segments,
    },
    markedPendingIds,
  };
}

function probeBrowserMediaDuration(objectUrl: string, isAudioOnly: boolean): Promise<number> {
  return new Promise((resolve) => {
    const el = document.createElement(isAudioOnly ? "audio" : "video");
    el.preload = "metadata";
    el.onloadedmetadata = () => {
      const d = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : 15;
      resolve(Number(d.toFixed(2)));
    };
    el.onerror = () => resolve(15);
    el.src = objectUrl;
  });
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  script: null,
  mode: "video_dubbing",
  selectedSegmentId: null,
  currentTime: 0,
  isPlaying: false,
  isLoading: false,
  isSynthesizing: false,
  isDirty: false,
  error: null,
  lastSyncMessage: null,
  backendStatus: "connecting",
  gpuBusy: false,
  projectList: [],
  trackStates: { ...DEFAULT_TRACK_STATES },
  timelineZoom: 72,
  audioVersionMap: {},
  pastScripts: [],
  futureScripts: [],
  lastUndoMergeKey: null,

  initWorkspace: async () => {
    set({ backendStatus: "connecting", error: null });
    try {
      const health = await checkHealth();
      const projects = await fetchProjects().catch(() => []);
      set({
        backendStatus: "online",
        gpuBusy: health.busy,
        projectList: projects,
        lastSyncMessage: "后端服务在线，等待选择或创建项目",
      });
    } catch {
      set({
        backendStatus: "offline",
        gpuBusy: false,
        projectList: [],
        error: null,
        lastSyncMessage: "离线模式：未连接后端",
      });
    }
  },

  closeProject: () =>
    set({
      script: null,
      selectedSegmentId: null,
      currentTime: 0,
      isPlaying: false,
      isDirty: false,
      pastScripts: [],
      futureScripts: [],
      lastUndoMergeKey: null,
      error: null,
      lastSyncMessage: null,
    }),

  importMediaFile: async (file) => {
    set({ isLoading: true, error: null });
    const isAudio = file.type.startsWith("audio/");
    const targetMode: ProjectMode = isAudio ? "virtual_stage" : "video_dubbing";

    try {
      const uploaded = await uploadProjectMedia(file, targetMode);
      const created = await fetchProjectScript(uploaded.project_id);
      const projects = await fetchProjects().catch(() => [uploaded.project_id]);
      set({
        script: created,
        mode: created.mode,
        selectedSegmentId: created.segments[0]?.id ?? null,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        pastScripts: [],
        futureScripts: [],
        lastUndoMergeKey: null,
        backendStatus: "online",
        projectList: projects,
        lastSyncMessage: `已上传并初始化项目 ${uploaded.project_id}`,
      });
      return;
    } catch {
      // 后端离线时构建前端临时本地可预览项目
      const objectUrl = URL.createObjectURL(file);
      const duration = await probeBrowserMediaDuration(objectUrl, isAudio);
      const cleanName =
        file.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") || `local_${Date.now()}`;
      const localScript: ProjectScript = {
        project_id: cleanName,
        mode: targetMode,
        media: {
          source_video: isAudio ? null : objectUrl,
          vocals_wav: isAudio ? objectUrl : null,
          bgm_wav: null,
          output_video: null,
          duration,
          bgm_volume: 0.75,
          auto_ducking: true,
        },
        glossary: {},
        characters: {
          SPEAKER_00: {
            display_name: "主讲人",
            speaking_style: "",
            color: "#5B8DEF",
            gpt_weights: null,
            sovits_weights: null,
            emotion_refs: {},
            avatar: { type: "live2d", model_url: "/assets/models/host_a/model3.json" },
          },
        },
        segments: [
          {
            id: 0,
            start: 0,
            end: Math.min(duration, 4.0),
            speaker: "SPEAKER_00",
            orig_text: "",
            words: [],
            alignment_needs_review: true,
      alignment_granularity: "unknown",
      timing_source: "unknown",
      segment_timing_needs_review: true,
            speaker_needs_review: true,
            target_text: "请点击“精修”编辑本句台词，或连接后端自动识别分轨。",
            target_lang: "zh",
            emotion: "neutral",
            action: "nod",
            keep_original: false,
            speed_factor: 1.0,
            tts_wav: null,
            actual_duration: null,
            status: "pending",
          },
        ],
      };

      set({
        script: localScript,
        mode: targetMode,
        selectedSegmentId: 0,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        pastScripts: [],
        futureScripts: [],
        lastUndoMergeKey: null,
        lastSyncMessage: `已导入本地媒体 ${file.name}（离线预览）`,
      });
    }
  },

  setScript: (script) =>
    set((state) => {
      const keepSelection =
        state.script?.project_id === script.project_id &&
        state.selectedSegmentId != null &&
        script.segments.some((seg) => seg.id === state.selectedSegmentId);
      const isDifferentProject = state.script?.project_id !== script.project_id;
      return {
        script,
        mode: script.mode,
        selectedSegmentId: keepSelection
          ? state.selectedSegmentId
          : (script.segments[0]?.id ?? null),
        error: null,
        isDirty: false,
        ...(isDifferentProject
          ? { pastScripts: [], futureScripts: [], lastUndoMergeKey: null }
          : { futureScripts: [], lastUndoMergeKey: null }),
      };
    }),

  setMode: (mode) =>
    set((state) => ({
      mode,
      isDirty: Boolean(state.script),
      script: state.script ? { ...state.script, mode } : null,
      ...(state.script
        ? {
            pastScripts: pushPastScript(state.pastScripts, state.script),
            futureScripts: [],
            lastUndoMergeKey: null,
          }
        : {}),
    })),

  selectSegment: (segmentId) => set({ selectedSegmentId: segmentId }),

  stepSegment: (delta) =>
    set((state) => {
      const segments = state.script?.segments ?? [];
      if (segments.length === 0) {
        return state;
      }
      const currentIdx = segments.findIndex((seg) => seg.id === state.selectedSegmentId);
      const nextIdx =
        currentIdx === -1 ? 0 : Math.max(0, Math.min(segments.length - 1, currentIdx + delta));
      const target = segments[nextIdx];
      return {
        selectedSegmentId: target.id,
        currentTime: target.start,
      };
    }),

  setCurrentTime: (time) =>
    set((state) => {
      const maxDuration = state.script?.media.duration ?? 30;
      const clamped = Math.max(0, Math.min(maxDuration, time));
      return { currentTime: clamped };
    }),

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  setTimelineZoom: (zoom) => set({ timelineZoom: Math.max(30, Math.min(180, Math.round(zoom))) }),

  toggleTrackState: (key) =>
    set((state) => ({
      trackStates: {
        ...state.trackStates,
        [key]: !state.trackStates[key],
      },
    })),

  bumpAudioVersion: (segmentIds) =>
    set((state) => {
      const now = Date.now();
      const next = { ...state.audioVersionMap };
      for (const id of segmentIds) {
        next[id] = now;
      }
      return { audioVersionMap: next };
    }),

  commitHistoryCheckpoint: () =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      return {
        pastScripts: pushPastScript(state.pastScripts, state.script),
        futureScripts: [],
        lastUndoMergeKey: null,
      };
    }),

  undo: () => {
    const state = get();
    if (!state.script || state.pastScripts.length === 0) {
      return false;
    }
    const prev = state.pastScripts[state.pastScripts.length - 1];
    const nextPast = state.pastScripts.slice(0, -1);
    const keepSelection =
      state.selectedSegmentId != null &&
      prev.segments.some((seg) => seg.id === state.selectedSegmentId);
    set({
      script: prev,
      mode: prev.mode,
      selectedSegmentId: keepSelection ? state.selectedSegmentId : (prev.segments[0]?.id ?? null),
      pastScripts: nextPast,
      futureScripts: [state.script, ...state.futureScripts].slice(0, MAX_HISTORY_STEPS),
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: "已撤销上一步修改",
    });
    return true;
  },

  redo: () => {
    const state = get();
    if (!state.script || state.futureScripts.length === 0) {
      return false;
    }
    const next = state.futureScripts[0];
    const remainingFuture = state.futureScripts.slice(1);
    const keepSelection =
      state.selectedSegmentId != null &&
      next.segments.some((seg) => seg.id === state.selectedSegmentId);
    set({
      script: next,
      mode: next.mode,
      selectedSegmentId: keepSelection ? state.selectedSegmentId : (next.segments[0]?.id ?? null),
      pastScripts: pushPastScript(state.pastScripts, state.script),
      futureScripts: remainingFuture,
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: "已重做修改",
    });
    return true;
  },

  updateSegmentLocal: (patch, options) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      const { nextScript } = applyPatchToSegments(state.script, [patch]);
      if (options?.skipHistory) {
        return {
          script: nextScript,
          isDirty: true,
        };
      }
      const fieldKeys = Object.keys(patch)
        .filter((k) => k !== "id")
        .sort()
        .join(",");
      const mergeKey = `seg:${patch.id}:${fieldKeys}`;
      const now = Date.now();
      const canCoalesce =
        state.lastUndoMergeKey !== null &&
        state.lastUndoMergeKey.key === mergeKey &&
        now - state.lastUndoMergeKey.ts < UNDO_MERGE_WINDOW_MS;
      return {
        script: nextScript,
        isDirty: true,
        pastScripts: canCoalesce
          ? state.pastScripts
          : pushPastScript(state.pastScripts, state.script),
        futureScripts: [],
        lastUndoMergeKey: { key: mergeKey, ts: now },
      };
    }),

  batchUpdateSegmentsLocal: (patches) => {
    const state = get();
    if (!state.script || patches.length === 0) {
      return [];
    }
    const { nextScript, markedPendingIds } = applyPatchToSegments(state.script, patches);
    set({
      script: nextScript,
      isDirty: true,
      pastScripts: pushPastScript(state.pastScripts, state.script),
      futureScripts: [],
      lastUndoMergeKey: null,
    });
    return markedPendingIds;
  },

  splitSegment: (segmentId, splitTime) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return null;
    }
    const seg = script.segments.find((item) => item.id === segmentId);
    if (!seg || seg.end - seg.start < 0.4) {
      set({ lastSyncMessage: "片段时长不足 0.4s，无法分割" });
      return null;
    }
    const candidate =
      splitTime !== undefined && splitTime > seg.start + 0.15 && splitTime < seg.end - 0.15
        ? Number(splitTime.toFixed(2))
        : Number(((seg.start + seg.end) / 2).toFixed(2));
    if (candidate <= seg.start + 0.1 || candidate >= seg.end - 0.1) {
      set({ lastSyncMessage: "请将播放头置于片段内部后再分割" });
      return null;
    }

    const ratio = (candidate - seg.start) / Math.max(0.1, seg.end - seg.start);
    const [leftTarget, rightTarget] = splitTextNearRatio(seg.target_text, ratio);
    const getWordMid = (w: (typeof seg.words)[number]): number | null => {
      if (w.start != null && w.end != null) return (w.start + w.end) / 2;
      if (w.start != null) return w.start;
      if (w.end != null) return w.end;
      return null;
    };
    const rawSplitIdx = seg.words.findIndex((w) => {
      const mid = getWordMid(w);
      return mid !== null && mid >= candidate;
    });
    const splitIdx =
      rawSplitIdx !== -1
        ? rawSplitIdx
        : Math.max(1, Math.min(seg.words.length - 1, Math.round(seg.words.length * ratio)));
    const leftWords = seg.words.slice(0, splitIdx);
    const rightWords = seg.words.slice(splitIdx);
    const [leftOrig, rightOrig] = splitOrigTextByWords(seg.orig_text, seg.words, splitIdx, ratio);

    const newId = Math.max(-1, ...script.segments.map((item) => item.id)) + 1;
    const leftSeg = invalidateAudio({
      ...seg,
      end: candidate,
      orig_text: leftOrig,
      target_text: leftTarget || seg.target_text,
      words: leftWords,
      alignment_needs_review: true,
    });
    const rightSeg = invalidateAudio({
      ...seg,
      id: newId,
      start: candidate,
      orig_text: rightOrig,
      target_text: rightTarget || "（续）",
      words: rightWords,
      alignment_needs_review: true,
    });
    const segments = script.segments
      .map((item) => (item.id === segmentId ? leftSeg : item))
      .concat(rightSeg)
      .sort((a, b) => a.start - b.start || a.id - b.id);

    set({
      script: { ...script, segments },
      selectedSegmentId: newId,
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已在 ${candidate.toFixed(2)}s 分割片段 #${segmentId}`,
    });
    return newId;
  },

  mergeSegmentWithNext: (segmentId) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return false;
    }
    const sorted = [...script.segments].sort((a, b) => a.start - b.start || a.id - b.id);
    const idx = sorted.findIndex((item) => item.id === segmentId);
    if (idx === -1 || idx >= sorted.length - 1) {
      set({ lastSyncMessage: "当前片段后方没有可合并的片段" });
      return false;
    }
    const curr = sorted[idx];
    const next = sorted[idx + 1];
    const merged = invalidateAudio({
      ...curr,
      start: Number(Math.min(curr.start, next.start).toFixed(2)),
      end: Number(Math.max(curr.end, next.end).toFixed(2)),
      orig_text: joinSegmentText(curr.orig_text, next.orig_text),
      target_text: joinSegmentText(curr.target_text, next.target_text),
      words: [...curr.words, ...next.words],
      alignment_needs_review: Boolean(curr.alignment_needs_review || next.alignment_needs_review),
      speaker_needs_review: Boolean(
        curr.speaker_needs_review || next.speaker_needs_review || curr.speaker !== next.speaker,
      ),
    });
    const segments = sorted
      .filter((item) => item.id !== next.id)
      .map((item) => (item.id === curr.id ? merged : item));

    set({
      script: { ...script, segments },
      selectedSegmentId: curr.id,
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已合并片段 #${curr.id} 与 #${next.id}`,
    });
    return true;
  },

  addSegment: (options) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return null;
    }
    const sorted = [...script.segments].sort((a, b) => a.start - b.start || a.id - b.id);
    const selectedSeg = sorted.find((s) => s.id === state.selectedSegmentId);
    let characters = script.characters;
    const speaker =
      options?.speaker ?? selectedSeg?.speaker ?? Object.keys(script.characters)[0] ?? "SPEAKER_00";
    if (!characters[speaker]) {
      characters = {
        ...characters,
        [speaker]: {
          display_name: speaker,
          speaking_style: "",
          color: "#5B8DEF",
          gpt_weights: null,
          sovits_weights: null,
          emotion_refs: {},
          avatar: null,
        },
      };
    }

    let start = Math.max(0, Number((options?.start ?? state.currentTime).toFixed(2)));
    for (const seg of sorted) {
      if (seg.start < start + 0.27 && seg.end > start - 0.05) {
        start = Number((seg.end + 0.05).toFixed(2));
      }
    }
    const following = sorted.find((seg) => seg.start > start);
    const defaultEnd =
      options?.end !== undefined
        ? Math.max(start + 0.2, Number(options.end.toFixed(2)))
        : Number(
            (following ? Math.min(start + 2.0, following.start - 0.02) : start + 2.0).toFixed(2),
          );
    const end = Math.max(start + 0.25, defaultEnd);
    const newId = Math.max(-1, ...script.segments.map((item) => item.id)) + 1;
    const newSeg: Segment = {
      id: newId,
      start,
      end,
      speaker,
      orig_text: "",
      alignment_granularity: "unknown",
      timing_source: "unknown",
      segment_timing_needs_review: true,
      words: [],
      alignment_needs_review: true,
      speaker_needs_review: false,
      target_text: "新台词片段",
      target_lang: selectedSeg?.target_lang ?? "zh",
      emotion: "neutral",
      action: null,
      keep_original: false,
      speed_factor: 1.0,
      tts_wav: null,
      applied_speed_factor: null,
      actual_duration: null,
      status: "pending",
    };

    const segments = [...script.segments, newSeg].sort((a, b) => a.start - b.start || a.id - b.id);
    set({
      script: {
        ...script,
        characters,
        media: {
          ...script.media,
          duration: Math.max(script.media.duration, end),
        },
        segments,
      },
      selectedSegmentId: newId,
      currentTime: start,
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已在 ${start.toFixed(2)}s 新增片段 #${newId}`,
    });
    return newId;
  },

  deleteSegment: (segmentId) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return false;
    }
    const idx = script.segments.findIndex((item) => item.id === segmentId);
    if (idx === -1) {
      return false;
    }
    const segments = script.segments.filter((item) => item.id !== segmentId);
    const nextSelected = segments[idx]?.id ?? segments[idx - 1]?.id ?? null;
    set({
      script: { ...script, segments },
      selectedSegmentId: nextSelected,
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已删除片段 #${segmentId}`,
    });
    return true;
  },

  confirmAllPendingReviews: (speakerKey?: string) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return 0;
    }
    let confirmed = 0;
    const segments = script.segments.map((seg) => {
      if (speakerKey && seg.speaker !== speakerKey) {
        return seg;
      }
      const needsTiming = seg.segment_timing_needs_review !== false;
      const needsAlign = seg.alignment_needs_review !== false;
      const needsSpeaker = seg.speaker_needs_review !== false && seg.speaker !== "UNKNOWN";
      if (needsTiming || needsAlign || needsSpeaker) {
        confirmed += 1;
        return {
          ...seg,
          segment_timing_needs_review: false,
          alignment_needs_review: false,
          speaker_needs_review: seg.speaker === "UNKNOWN" ? seg.speaker_needs_review : false,
        };
      }
      return seg;
    });
    if (confirmed === 0) {
      return 0;
    }
    const scopeLabel = speakerKey
      ? `${script.characters[speakerKey]?.display_name || speakerKey} 的 ${confirmed} 条`
      : `全部 ${confirmed} 条`;
    set({
      script: { ...script, segments },
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已一键确认 ${scopeLabel}待校对片段`,
    });
    return confirmed;
  },

  applyVoicePresetToCharacter: (speakerKey, preset) => {
    const state = get();
    const script = state.script;
    const previous = script?.characters[speakerKey];
    if (!script || !previous) {
      return;
    }
    const clonedRefs = Object.fromEntries(
      Object.entries(preset.emotion_refs).map(([emo, ref]) => [emo, { ...ref }]),
    ) as CharacterConfig["emotion_refs"];
    const character: CharacterConfig = {
      ...previous,
      speaking_style: preset.speaking_style,
      gpt_weights: preset.gpt_weights ?? null,
      sovits_weights: preset.sovits_weights ?? null,
      emotion_refs: clonedRefs,
    };
    const segments = script.segments.map((seg) =>
      seg.speaker === speakerKey && !seg.keep_original ? invalidateAudio(seg) : seg,
    );
    set({
      script: {
        ...script,
        characters: {
          ...script.characters,
          [speakerKey]: character,
        },
        segments,
      },
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
      isDirty: true,
      lastSyncMessage: `已将预设“${preset.name}”应用到 ${previous.display_name || speakerKey}`,
    });
  },

  restoreSnapshot: async (snapshotId) => {
    const state = get();
    const script = state.script;
    if (!script) {
      return false;
    }
    set({ isLoading: true, error: null });
    try {
      const restored = await apiRestoreProjectSnapshot(script.project_id, snapshotId);
      const current = get();
      if (current.script?.project_id !== script.project_id) {
        set({ isLoading: false });
        return false;
      }
      const prevScript = current.script;
      current.bumpAudioVersion(restored.segments.map((seg) => seg.id));
      set({
        script: restored,
        mode: restored.mode,
        selectedSegmentId: restored.segments[0]?.id ?? null,
        pastScripts: prevScript
          ? pushPastScript(current.pastScripts, prevScript)
          : current.pastScripts,
        futureScripts: [],
        lastUndoMergeKey: null,
        isLoading: false,
        isDirty: false,
        lastSyncMessage: `已回滚恢复快照 ${snapshotId}`,
      });
      return true;
    } catch (err) {
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "恢复历史快照失败",
      });
      return false;
    }
  },

  updateBgmVolume: (volume) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      const clamped = Math.max(0, Math.min(2, Number(volume.toFixed(2))));
      const now = Date.now();
      const mergeKey = "media:bgm_volume";
      const canCoalesce =
        state.lastUndoMergeKey !== null &&
        state.lastUndoMergeKey.key === mergeKey &&
        now - state.lastUndoMergeKey.ts < UNDO_MERGE_WINDOW_MS;
      return {
        isDirty: true,
        script: {
          ...state.script,
          media: {
            ...state.script.media,
            bgm_volume: clamped,
          },
        },
        pastScripts: canCoalesce
          ? state.pastScripts
          : pushPastScript(state.pastScripts, state.script),
        futureScripts: [],
        lastUndoMergeKey: { key: mergeKey, ts: now },
      };
    }),

  updateStemVolume: (stem, volume) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      const clamped = Math.max(0, Math.min(2, Number(volume.toFixed(2))));
      const current = state.script.media.stem_volumes ?? {
        singing: 1,
        music: 1,
        effects: 1,
      };
      const now = Date.now();
      const mergeKey = `media:stem:${stem}`;
      const canCoalesce =
        state.lastUndoMergeKey !== null &&
        state.lastUndoMergeKey.key === mergeKey &&
        now - state.lastUndoMergeKey.ts < UNDO_MERGE_WINDOW_MS;
      return {
        isDirty: true,
        script: {
          ...state.script,
          media: {
            ...state.script.media,
            stem_volumes: {
              ...current,
              [stem]: clamped,
            },
          },
        },
        pastScripts: canCoalesce
          ? state.pastScripts
          : pushPastScript(state.pastScripts, state.script),
        futureScripts: [],
        lastUndoMergeKey: { key: mergeKey, ts: now },
      };
    }),

  updateAutoDucking: (enabled) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      return {
        isDirty: true,
        script: {
          ...state.script,
          media: {
            ...state.script.media,
            auto_ducking: enabled,
          },
        },
        pastScripts: pushPastScript(state.pastScripts, state.script),
        futureScripts: [],
        lastUndoMergeKey: null,
      };
    }),

  updateCharacter: (speakerKey, patch) =>
    set((state) => {
      const script = state.script;
      const previous = script?.characters[speakerKey];
      if (!script || !previous) {
        return state;
      }
      const character = { ...previous, ...patch };
      const voiceChanged =
        character.gpt_weights !== previous.gpt_weights ||
        character.sovits_weights !== previous.sovits_weights ||
        character.emotion_refs !== previous.emotion_refs;
      const fieldKeys = Object.keys(patch).sort().join(",");
      const mergeKey = `char:${speakerKey}:${fieldKeys}`;
      const now = Date.now();
      const canCoalesce =
        state.lastUndoMergeKey !== null &&
        state.lastUndoMergeKey.key === mergeKey &&
        now - state.lastUndoMergeKey.ts < UNDO_MERGE_WINDOW_MS;
      return {
        isDirty: true,
        script: {
          ...script,
          characters: {
            ...script.characters,
            [speakerKey]: character,
          },
          segments: voiceChanged
            ? script.segments.map((seg) =>
                seg.speaker === speakerKey && !seg.keep_original ? invalidateAudio(seg) : seg,
              )
            : script.segments,
        },
        pastScripts: canCoalesce ? state.pastScripts : pushPastScript(state.pastScripts, script),
        futureScripts: [],
        lastUndoMergeKey: { key: mergeKey, ts: now },
      };
    }),

  addCharacter: (name) => {
    const state = get();
    const script = state.script;
    const displayName = name.trim();
    if (!script || !displayName) return null;
    let index = Object.keys(script.characters).length + 1;
    while (script.characters[`character_${index}`]) index += 1;
    const speaker = `character_${index}`;
    set({
      isDirty: true,
      script: {
        ...script,
        characters: {
          ...script.characters,
          [speaker]: {
            display_name: displayName,
            speaking_style: "",
            color: "#3b82f6",
            gpt_weights: null,
            sovits_weights: null,
            emotion_refs: {},
            avatar: null,
          },
        },
      },
      pastScripts: pushPastScript(state.pastScripts, script),
      futureScripts: [],
      lastUndoMergeKey: null,
    });
    return speaker;
  },

  mergeCharacters: (source, destination) =>
    set((state) => {
      const script = state.script;
      if (!script?.characters[source] || !script.characters[destination] || source === destination)
        return state;
      const patches = script.segments
        .filter((seg) => seg.speaker === source)
        .map((seg) => ({
          id: seg.id,
          speaker: destination,
          speaker_needs_review: seg.speaker_needs_review ?? true,
        }));
      const { nextScript } = applyPatchToSegments(script, patches);
      const characters = { ...nextScript.characters };
      delete characters[source];
      return {
        isDirty: true,
        script: { ...nextScript, characters },
        pastScripts: pushPastScript(state.pastScripts, script),
        futureScripts: [],
        lastUndoMergeKey: null,
      };
    }),

  updateTranslationSettings: (sourceLang, glossary) =>
    set((state) =>
      state.script
        ? {
            script: { ...state.script, source_lang: sourceLang, glossary },
            isDirty: true,
            pastScripts: pushPastScript(state.pastScripts, state.script),
            futureScripts: [],
            lastUndoMergeKey: null,
          }
        : state,
    ),

  retranslate: async () => {
    const { script, isDirty, backendStatus, isLoading, isSynthesizing } = get();
    if (!script || backendStatus !== "online") return "请连接后端后重译";
    if (isLoading || isSynthesizing) return "请等待当前操作完成";
    set({ isLoading: true, error: null });
    try {
      if (isDirty) await apiSaveProjectScript(script);
      const result = await translateProject(script.project_id);
      if (get().script !== script) return "重译已在后端完成，当前编辑保留在本地";
      const errNote = result.errors?.length ? `（部分批次失败：${result.errors[0]}）` : "";
      const message = `已翻译 ${result.translated_ids.length} 条，保留 ${result.skipped_count} 条${errNote}`;
      set({ script: result.script, isDirty: false, lastSyncMessage: message });
      return message;
    } catch (err) {
      return err instanceof Error ? err.message : "重译失败";
    } finally {
      set({ isLoading: false });
    }
  },

  loadProjectScript: async (projectId) => {
    set({ isLoading: true, error: null });
    try {
      const data = await fetchProjectScript(projectId);
      set({
        script: data,
        mode: data.mode,
        selectedSegmentId: data.segments[0]?.id ?? null,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        pastScripts: [],
        futureScripts: [],
        lastUndoMergeKey: null,
        backendStatus: "online",
        lastSyncMessage: `已加载项目 ${projectId}`,
      });
    } catch (err) {
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "加载剧本失败",
      });
    }
  },

  saveProjectScript: async () => {
    const { script, backendStatus } = get();
    if (!script) {
      return;
    }
    if (backendStatus === "offline") {
      set({
        isDirty: false,
        lastSyncMessage: "已保存至本地会话",
      });
      return;
    }
    set({ isLoading: true, error: null });
    try {
      const saved = await apiSaveProjectScript(script);
      if (get().script !== script) {
        set({ isLoading: false });
        return;
      }
      set({
        script: saved,
        isLoading: false,
        isDirty: false,
        backendStatus: "online",
        lastSyncMessage: "已保存至后端剧本文件",
      });
    } catch (err) {
      if (err instanceof ApiError && err.isNetworkError) {
        set({
          isLoading: false,
          backendStatus: "offline",
          isDirty: false,
          lastSyncMessage: "后端暂不可达，已暂存本地",
        });
        return;
      }
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "保存失败",
      });
    }
  },

  synthesizeSegment: async (segmentIds) => {
    const state = get();
    const { script, backendStatus } = state;
    if (!script) {
      return { simulated: true, message: "请先打开项目" };
    }

    const targetIds =
      segmentIds && segmentIds.length > 0
        ? segmentIds
        : script.segments
            .filter((seg) => !seg.keep_original && seg.status !== "ready")
            .map((seg) => seg.id);

    if (targetIds.length === 0) {
      return { simulated: false, message: "没有待合成的片段" };
    }

    const applyLocalSimulatedSynth = (msg: string) => {
      const targetSet = new Set(targetIds);
      const latestScript = get().script;
      if (!latestScript) {
        set({ isSynthesizing: false });
        return { simulated: true, message: "无有效剧本" };
      }

      const nextSegments = latestScript.segments.map((seg) => {
        if (!targetSet.has(seg.id) || seg.keep_original) {
          return seg;
        }
        const slot = Math.max(0.4, seg.end - seg.start);
        const estimated = Math.min(
          slot * 1.12,
          Math.max(0.8, seg.target_text.length / 4.8 / Math.max(0.5, seg.speed_factor)),
        );
        const paddedId = String(seg.id).padStart(4, "0");
        return {
          ...seg,
          tts_wav: seg.tts_wav ?? `/workspace/${latestScript.project_id}/tts/seg_${paddedId}.wav`,
          actual_duration: Number(estimated.toFixed(2)),
          status: "ready" as const,
        };
      });

      get().bumpAudioVersion(targetIds);
      set({
        script: { ...latestScript, segments: nextSegments },
        isSynthesizing: false,
        lastSyncMessage: `演示合成完成（#${targetIds.join(", #")}）`,
      });
      return {
        simulated: true,
        message: msg,
      };
    };

    if (backendStatus === "offline") {
      return applyLocalSimulatedSynth("离线模式：已完成演示合成");
    }

    set({ isSynthesizing: true, error: null });
    try {
      if (state.isDirty) {
        const saved = await apiSaveProjectScript(script);
        if (get().script?.project_id !== script.project_id) {
          set({ isSynthesizing: false });
          return { simulated: false, message: "项目已切换，已取消合成" };
        }
        set((curr) => (curr.script === script ? { script: saved, isDirty: false } : {}));
      }
      const res = await apiSynthesizeSegments(script.project_id, targetIds);
      if (get().script?.project_id !== script.project_id) {
        set({ isSynthesizing: false });
        return { simulated: false, message: "项目已切换，合成结果已保存到原项目" };
      }
      const issues = synthesisIssues(res);
      const message = `已合成 ${res.synthesized_ids.length} 条${issues.length ? ` · ${issues[0]}` : ""}`;
      get().bumpAudioVersion(res.synthesized_ids);
      set((curr) => {
        const keepSelection =
          curr.selectedSegmentId != null &&
          res.script.segments.some((seg) => seg.id === curr.selectedSegmentId);
        return {
          script: res.script,
          selectedSegmentId: keepSelection
            ? curr.selectedSegmentId
            : (res.script.segments[0]?.id ?? null),
          isSynthesizing: false,
          isDirty: false,
          backendStatus: "online",
          lastSyncMessage: message,
        };
      });
      return {
        simulated: false,
        message,
        issues,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : "合成失败";
      set({ isSynthesizing: false, error: message });
      return { simulated: false, message, issues: [message] };
    }
  },
}));
