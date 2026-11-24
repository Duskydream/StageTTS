import {
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import { create } from "zustand";
import {
  type ParseStatusResponse,
  fetchParseStatus,
  parseProjectMedia,
  renderVideo,
  saveProjectScript,
  subscribeParseStatus,
  synthesizeSegments,
  translateProject,
} from "../services/api";
import type { ProjectScript } from "../types/script";
import type {
  CenterViewTab,
  LeftDrawerTab,
  NodeRunStatus,
  PipelineConfig,
  PipelineLogEntry,
  RightPanelTab,
  StageNodeData,
  WorkspacePreset,
} from "../types/workflow";
import { pipelineStartForNode, pipelineNodeHint, type PipelineStartStep } from "../types/workflow";
import { synthesisIssues } from "../utils/synthesisIssues";
import { useProjectStore } from "./useProjectStore";

export type StageFlowNode = Node<StageNodeData, "stageNode">;

function createInitialNodes(config: PipelineConfig): StageFlowNode[] {
  return [
    {
      id: "media_input",
      type: "stageNode",
      position: { x: 20, y: 120 },
      data: {
        title: "媒体输入",
        category: "input",
        env: "env_backend",
        vramGb: 0,
        status: "idle",
        inputs: [],
        outputs: [{ id: "out-video", label: "媒体", type: "MEDIA" }],
        summary: "",
        params: { format: "mp4 / wav" },
      },
    },
    {
      id: "banquet",
      type: "stageNode",
      position: { x: 300, y: 20 },
      data: {
        title: "音轨分离",
        category: "parse",
        env: "env_parse",
        vramGb: 1.8,
        status: "idle",
        inputs: [{ id: "in-media", label: "媒体流", type: "MEDIA" }],
        outputs: [
          { id: "out-vocals", label: "对白", type: "AUDIO" },
          { id: "out-bgm", label: "背景", type: "AUDIO" },
        ],
        summary: "Banquet 4-Stem",
        params: { model: "banquet" },
      },
    },
    {
      id: "whisperx",
      type: "stageNode",
      position: { x: 580, y: 20 },
      data: {
        title: "MOSS 识别",
        category: "parse",
        env: "env_parse",
        vramGb: 2.0,
        status: "idle",
        inputs: [{ id: "in-vocals", label: "对白", type: "AUDIO" }],
        outputs: [{ id: "out-turns", label: "句段", type: "SEGMENTS" }],
        summary: "BF16 · ≤90m",
        params: { model: "OpenMOSS-Team/MOSS-Transcribe-Diarize", precision: "bfloat16", max_audio_seconds: 5400 },
      },
    },
    {
      id: "llm_tagger",
      type: "stageNode",
      position: { x: 860, y: 20 },
      data: {
        title: "翻译与情绪",
        category: "parse",
        env: "env_parse",
        vramGb: 1.1,
        status: "idle",
        inputs: [{ id: "in-turns", label: "句段", type: "SEGMENTS" }],
        outputs: [{ id: "out-script-draft", label: "初稿", type: "JSON" }],
        summary: "",
        params: { target_lang: config.targetLang },
      },
    },
    {
      id: "script_hub",
      type: "stageNode",
      position: { x: 580, y: 230 },
      data: {
        title: "剧本枢纽",
        category: "hub",
        env: "env_backend",
        vramGb: 0,
        status: "idle",
        inputs: [
          { id: "in-draft", label: "初稿", type: "JSON" },
          { id: "in-bgm", label: "背景", type: "AUDIO" },
        ],
        outputs: [
          { id: "out-script", label: "剧本", type: "SCRIPT" },
          { id: "out-bgm-pass", label: "背景", type: "AUDIO" },
        ],
        summary: "",
        params: { schema: "v2" },
      },
    },
    {
      id: "agent_tts",
      type: "stageNode",
      position: { x: 860, y: 230 },
      data: {
        title: "语音合成",
        category: "synthesis",
        env: "env_tts",
        vramGb: 4.2,
        status: "idle",
        inputs: [{ id: "in-script", label: "剧本", type: "SCRIPT" }],
        outputs: [{ id: "out-wavs", label: "配音", type: "AUDIO[]" }],
        summary: "GPT-SoVITS",
        params: { overflow_ratio: 1.15, max_rounds: 2 },
      },
    },
    {
      id: "ffmpeg_mixer",
      type: "stageNode",
      position: { x: 1140, y: 110 },
      data: {
        title: "混音导出",
        category: "output",
        env: "env_backend",
        vramGb: 0,
        status: "idle",
        inputs: [
          { id: "in-wavs", label: "配音", type: "AUDIO[]" },
          { id: "in-bgm", label: "背景", type: "AUDIO" },
        ],
        outputs: [{ id: "out-mp4", label: "成片", type: "MP4" }],
        summary: "",
        params: { burn_watermark: config.burnWatermark },
      },
    },
  ];
}

const INITIAL_EDGES: Edge[] = [
  {
    id: "e-input-banquet",
    source: "media_input",
    sourceHandle: "out-video",
    target: "banquet",
    targetHandle: "in-media",
    style: { stroke: "#A87B70", strokeWidth: 1.5 },
  },
  {
    id: "e-banquet-whisper",
    source: "banquet",
    sourceHandle: "out-vocals",
    target: "whisperx",
    targetHandle: "in-vocals",
    style: { stroke: "#927A9E", strokeWidth: 1.5 },
  },
  {
    id: "e-whisper-llm",
    source: "whisperx",
    sourceHandle: "out-turns",
    target: "llm_tagger",
    targetHandle: "in-turns",
    style: { stroke: "#927A9E", strokeWidth: 1.5 },
  },
  {
    id: "e-llm-hub",
    source: "llm_tagger",
    sourceHandle: "out-script-draft",
    target: "script_hub",
    targetHandle: "in-draft",
    style: { stroke: "#AD925F", strokeWidth: 1.5 },
  },
  {
    id: "e-banquet-hub-bgm",
    source: "banquet",
    sourceHandle: "out-bgm",
    target: "script_hub",
    targetHandle: "in-bgm",
    style: { stroke: "#5C5C5C", strokeWidth: 1.25, strokeDasharray: "4 4" },
  },
  {
    id: "e-hub-tts",
    source: "script_hub",
    sourceHandle: "out-script",
    target: "agent_tts",
    targetHandle: "in-script",
    style: { stroke: "#5C9186", strokeWidth: 1.5 },
  },
  {
    id: "e-tts-mixer",
    source: "agent_tts",
    sourceHandle: "out-wavs",
    target: "ffmpeg_mixer",
    targetHandle: "in-wavs",
    style: { stroke: "#698EAD", strokeWidth: 1.5 },
  },
  {
    id: "e-hub-mixer-bgm",
    source: "script_hub",
    sourceHandle: "out-bgm-pass",
    target: "ffmpeg_mixer",
    targetHandle: "in-bgm",
    style: { stroke: "#5C5C5C", strokeWidth: 1.25, strokeDasharray: "4 4" },
  },
];

function nowTimeStr(): string {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function makeLog(
  env: PipelineLogEntry["env"],
  message: string,
  level: PipelineLogEntry["level"] = "info",
): PipelineLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: nowTimeStr(),
    env,
    message,
    level,
  };
}

const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  targetLang: "zh",
  burnWatermark: false,
  burnSubtitles: "none",
};

export function workflowPreflight(script: ProjectScript, config: PipelineConfig & { startStep: PipelineStartStep }, parseOnly: boolean): string | null {
  if (parseOnly && config.startStep !== "parse" && config.startStep !== "translate") {
    return "仅生成剧本支持从媒体解析或翻译开始；合成/导出请使用「运行后续管线」。";
  }
  if (config.startStep === "parse") {
    if (!script.media.source_video && !script.media.vocals_wav) return "项目缺少输入媒体，请先导入音视频。";
    return null;
  }
  if (!script.segments.length) return "项目没有对白片段，请先从媒体解析开始。";
  if (config.startStep === "translate" && script.segments.some((seg) => !seg.orig_text.trim())) {
    return "存在空白源台词，请补齐原文后翻译已有对白。";
  }
  if (config.startStep === "synthesize") {
    const empty = script.segments.filter((seg) => !seg.keep_original && !seg.target_text.trim());
    if (empty.length) return `片段 ${empty.map((seg) => seg.id).join("、")} 缺少译文，请先翻译或填写目标台词。`;
    const unchanged = script.segments.filter((seg) => !seg.keep_original && seg.target_text.trim() === seg.orig_text.trim()
      && (!script.source_lang || seg.target_lang !== script.source_lang));
    if (unchanged.length) return `片段 ${unchanged.map((seg) => seg.id).join("、")} 仍为源台词，无法确认已翻译；请先从翻译开始。`;
    if (!parseOnly && script.mode === "video_dubbing" && !script.media.source_video) return "缺少源视频，无法执行后续成片导出。";
  }
  if (config.startStep === "render") return renderPreflight(script);
  if (!parseOnly && script.mode === "video_dubbing" && !script.media.source_video) return "缺少源视频；可选择「仅生成剧本」完成翻译。";
  return null;
}

function renderPreflight(script: ProjectScript): string | null {
  if (script.mode === "video_dubbing" && !script.media.source_video) return "缺少源视频，无法导出成片。";
  if (script.segments.some((seg) => seg.keep_original) && !script.media.vocals_wav) return "保留原声片段缺少对白音轨，无法导出。";
  const missing = script.segments.filter((seg) => !seg.keep_original && (seg.status !== "ready" || !seg.tts_wav));
  if (missing.length) return `片段 ${missing.map((seg) => seg.id).join("、")} 配音尚未就绪，已阻止导出；请先完成语音合成。`;
  return null;
}

interface WorkflowState {
  workspacePreset: WorkspacePreset;
  centerTab: CenterViewTab;
  rightTab: RightPanelTab;
  leftDrawer: LeftDrawerTab;
  selectedNodeId: string | null;
  nodes: StageFlowNode[];
  edges: Edge[];
  pipelineConfig: PipelineConfig;
  isPipelineRunning: boolean;
  isPauseRequested: boolean;
  isPipelinePaused: boolean;
  isStopRequested: boolean;
  pausePipeline: () => void;
  resumePipeline: () => void;
  stopPipeline: () => void;
  pipelineLogs: PipelineLogEntry[];

  setWorkspacePreset: (preset: WorkspacePreset) => void;
  setCenterTab: (tab: CenterViewTab) => void;
  setRightTab: (tab: RightPanelTab) => void;
  showWorkflowLogs: boolean;
  toggleWorkflowLogs: (open?: boolean) => void;
  toggleLeftDrawer: (tab: Exclude<LeftDrawerTab, "none">) => void;
  selectNode: (nodeId: string | null) => void;
  onNodesChange: (changes: NodeChange<StageFlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  updatePipelineConfig: (patch: Partial<PipelineConfig>) => void;
  resetGraphLayout: () => void;
  appendLog: (
    env: PipelineLogEntry["env"],
    message: string,
    level?: PipelineLogEntry["level"],
  ) => void;
  runPipeline: (options?: { parseOnly?: boolean }) => Promise<void>;
  runPipelineSimulation: () => Promise<void>;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// A suspended run keeps its local script/config snapshot and waits for a user
// action without polling. Only one real workflow may own this continuation.
let releasePipelinePause: (() => void) | null = null;

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  workspacePreset: "editing",
  centerTab: "video",
  rightTab: "inspector",
  leftDrawer: "none",
  showWorkflowLogs: false,
  selectedNodeId: null,
  nodes: createInitialNodes(DEFAULT_PIPELINE_CONFIG),
  edges: INITIAL_EDGES,
  pipelineConfig: DEFAULT_PIPELINE_CONFIG,
  isPipelineRunning: false,
  isPauseRequested: false,
  isPipelinePaused: false,
  isStopRequested: false,
  pausePipeline: () => {
    if (!get().isPipelineRunning || get().isStopRequested || get().isPauseRequested) return;
    set({ isPauseRequested: true });
    get().appendLog("system", "已请求暂停：等待当前步骤完成并保存后暂停，不再启动下一步骤。", "info");
  },
  resumePipeline: () => {
    if (!get().isPipelineRunning || !get().isPauseRequested || get().isStopRequested) return;
    const wasPaused = get().isPipelinePaused;
    set({ isPauseRequested: false, isPipelinePaused: false });
    get().appendLog("system", wasPaused ? "继续执行剩余步骤。" : "已取消暂停请求，继续执行。", "info");
    releasePipelinePause?.();
  },
  stopPipeline: () => {
    if (!get().isPipelineRunning || get().isStopRequested) return;
    set({ isStopRequested: true, isPauseRequested: false, isPipelinePaused: false });
    get().appendLog("system", "已请求结束本次流程：保留已完成结果，当前请求结束后不再执行后续步骤。", "warn");
    releasePipelinePause?.();
  },
  pipelineLogs: [makeLog("system", "RTX 5060 8GB 串行显存接力策略已就绪", "info")],

  setWorkspacePreset: (preset) => {
    if (preset === "workflow") {
      set({
        workspacePreset: preset,
        centerTab: "graph",
      });
      return;
    }
    set((state) => ({
      workspacePreset: "editing",
      centerTab: "video",
      rightTab: state.rightTab === "logs" ? "inspector" : state.rightTab,
    }));
  },

  setCenterTab: (centerTab) => set({ centerTab }),

  setRightTab: (rightTab) => set({ rightTab }),

  toggleWorkflowLogs: (open) =>
    set((state) => ({
      showWorkflowLogs: open === undefined ? !state.showWorkflowLogs : open,
    })),

  toggleLeftDrawer: (tab) =>
    set((state) => ({
      leftDrawer: state.leftDrawer === tab ? "none" : tab,
    })),

  selectNode: (nodeId) =>
    set((state) => ({
      selectedNodeId: nodeId,
      nodes: state.nodes.map((node) => ({ ...node, selected: node.id === nodeId })),
    })),

  onNodesChange: (changes) =>
    set((state) => {
      const nodes = applyNodeChanges(changes, state.nodes);
      const selectionChanged = changes.some((change) => change.type === "select" || change.type === "remove");
      return { nodes, selectedNodeId: selectionChanged ? (nodes.find((node) => node.selected)?.id ?? null) : state.selectedNodeId };
    }),

  onEdgesChange: (changes) =>
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
    })),

  updatePipelineConfig: (patch) =>
    set((state) => {
      if (state.isPipelineRunning) return state;
      const nextConfig = { ...state.pipelineConfig, ...patch };
      const nodes = state.nodes.map((node) => {
        if (node.id === "whisperx") return { ...node, data: { ...node.data, vramGb: 2.0, summary: "BF16 · ≤90m" } };
        if (node.id === "llm_tagger") {
          return { ...node, data: { ...node.data, params: { ...node.data.params, target_lang: nextConfig.targetLang } } };
        }
        if (node.id === "ffmpeg_mixer") return { ...node, data: { ...node.data, params: { ...node.data.params, burn_watermark: nextConfig.burnWatermark } } };
        return node;
      });
      return { pipelineConfig: nextConfig, nodes };
    }),

  resetGraphLayout: () =>
    set((state) => state.isPipelineRunning ? state : ({
      nodes: createInitialNodes(state.pipelineConfig).map((node) => ({ ...node, selected: node.id === state.selectedNodeId })),
      edges: INITIAL_EDGES,
    })),

  appendLog: (env, message, level = "info") =>
    set((state) => ({
      pipelineLogs: [...state.pipelineLogs.slice(-39), makeLog(env, message, level)],
    })),

  runPipeline: async (options) => {
    const parseOnly = Boolean(options?.parseOnly);
    if (get().isPipelineRunning) {
      return;
    }

    const projectState = useProjectStore.getState();
    const currentScript = projectState.script;
    if (!currentScript) {
      set({ showWorkflowLogs: true });
      get().appendLog("system", "请先导入音视频或打开一个项目后再运行管线", "warn");
      return;
    }

    if (projectState.backendStatus !== "online") {
      set({ showWorkflowLogs: true });
      get().appendLog("system", "后端未连接，真实管线已阻止；请连接后端后重试。", "error");
      return;
    }
    if (projectState.isLoading || projectState.isSynthesizing) {
      get().appendLog("system", "项目正在加载、保存或合成，请等待当前操作完成。", "warn");
      return;
    }
    const startStep = pipelineStartForNode(get().selectedNodeId);
    if (!startStep) {
      set({ showWorkflowLogs: true });
      get().appendLog("system", pipelineNodeHint(get().selectedNodeId), "warn");
      return;
    }
    const cfg = { ...get().pipelineConfig, startStep };
    const preflight = workflowPreflight(currentScript, cfg, parseOnly);
    if (preflight) {
      set({ showWorkflowLogs: true });
      get().appendLog("system", `管线已阻止：${preflight}`, "error");
      return;
    }

    const setNodeStatus = (nodeId: string, status: NodeRunStatus) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, status } } : node,
        ),
      }));
    };

    const setActiveEdges = (activeEdgeIds: string[]) => {
      const activeSet = new Set(activeEdgeIds);
      set((state) => ({
        edges: state.edges.map((edge) => ({
          ...edge,
          animated: activeSet.has(edge.id),
        })),
      }));
    };

    const targetProjectId = currentScript.project_id;
    const isSameProject = () => useProjectStore.getState().script?.project_id === targetProjectId;
    const needsParse = cfg.startStep === "parse";
    const needsTranslate = cfg.startStep === "translate";
    const needsSynthesize = !parseOnly && cfg.startStep !== "render";
    const queued = new Set<string>([
      ...(needsParse ? ["media_input", "banquet", "whisperx", "llm_tagger", "script_hub"] : []),
      ...(needsTranslate ? ["llm_tagger", "script_hub"] : []),
      ...(!parseOnly ? ["ffmpeg_mixer"] : []),
      ...(needsSynthesize ? ["agent_tts"] : []),
    ]);
    set((state) => ({
      isPipelineRunning: true,
      isPauseRequested: false,
      isPipelinePaused: false,
      isStopRequested: false,
      nodes: state.nodes.map((node) => ({ ...node, data: { ...node.data, status: queued.has(node.id) ? "queued" : "idle" } })),
    }));
    let activeScript = currentScript;
    let activeNode = needsParse ? "media_input" : needsTranslate ? "llm_tagger" : needsSynthesize ? "agent_tts" : "ffmpeg_mixer";
    if (!needsParse) {
      get().appendLog("system", `从${needsTranslate ? "翻译" : needsSynthesize ? "语音合成" : "导出"}开始，使用已存源台词与设置；前序分离/识别未运行（节点保持待机）。`, "info");
    }

    // Pause only between requests: every completed backend step has already
    // persisted its result. Never abort fetch and pretend the worker stopped.
    const beforeNextStep = async (label: string): Promise<boolean> => {
      if (get().isPauseRequested && !get().isStopRequested) {
        setActiveEdges([]);
        const continuation = new Promise<void>((resolve) => { releasePipelinePause = resolve; });
        set({ isPipelinePaused: true });
        get().appendLog("system", `已暂停，下一步：${label}。可继续或结束本次流程。`, "info");
        await continuation;
        releasePipelinePause = null;
      }
      if (get().isStopRequested) {
        get().appendLog("system", "本次流程已结束，已完成的结果已保留。", "info");
        return false;
      }
      return true;
    };

    try {
      // Persist the current source text, glossary and character settings before
      // invoking any resume endpoint; never replace it with stale disk content.
      activeScript = await saveProjectScript(activeScript);
      if (isSameProject() && useProjectStore.getState().script === currentScript) {
        useProjectStore.getState().setScript(activeScript);
      }
      // -------------------------------------------------------------------
      // 阶段 A：调用真实后端 POST /api/projects/{id}/parse（env_parse 子进程）
      // -------------------------------------------------------------------
      if (needsParse) {
        if (!await beforeNextStep("媒体解析")) return;
        activeNode = "media_input";
        setNodeStatus("media_input", "running");
        setActiveEdges(["e-input-banquet"]);
        get().appendLog(
          "env_parse",
          `启动真实 4 阶段解析：项目 ${activeScript.project_id}（目标语种=${cfg.targetLang}，MOSS BF16/SDPA）`,
          "info",
        );

        let seenLogCount = 0;
        const applyStageFromWorker = (stage: string) => {
          if (stage === "Stage0" || stage === "Init") {
            activeNode = "media_input";
            setNodeStatus("media_input", "running");
            setActiveEdges(["e-input-banquet"]);
          } else if (stage === "Stage1") {
            activeNode = "banquet";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "running");
            setActiveEdges(["e-banquet-whisper", "e-banquet-hub-bgm"]);
          } else if (stage === "Stage2" || stage === "Stage3") {
            activeNode = "whisperx";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "running");
            setActiveEdges(["e-whisper-llm"]);
          } else if (stage === "Stage4") {
            activeNode = "llm_tagger";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "done");
            setNodeStatus("llm_tagger", "running");
            setActiveEdges(["e-llm-hub"]);
          } else if (stage === "Done") {
            activeNode = "script_hub";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "done");
            setNodeStatus("llm_tagger", "done");
            setNodeStatus("script_hub", "done");
          }
        };

        const applyParseStatus = (status: ParseStatusResponse, final = false) => {
          if (!final && !status.active) {
            return;
          }
          const count = status.log_count ?? status.logs.length;
          if (count < seenLogCount) {
            return;
          }
          if (status.stage && status.stage !== "idle") {
            applyStageFromWorker(status.stage);
          }
          const firstNew = Math.max(0, status.logs.length - (count - seenLogCount));
          for (const line of status.logs.slice(firstNew)) {
            const level = /使用未分离音频|失败|显存不足/.test(line) ? "warn" : "info";
            get().appendLog("env_parse", line, level);
          }
          seenLogCount = count;
        };

        const refreshParseStatus = async (final = false) => {
          try {
            const status = await fetchParseStatus(targetProjectId);
            applyParseStatus(status, final);
          } catch {
            // 解析仍由 POST 请求跟踪。
          }
        };

        let fallbackTimer: number | undefined;
        const unsubscribeSse = subscribeParseStatus(
          targetProjectId,
          (status) => applyParseStatus(status),
          () => {
            if (fallbackTimer === undefined) {
              fallbackTimer = window.setInterval(() => void refreshParseStatus(), 1200);
            }
          },
        );

        try {
          activeScript = await parseProjectMedia(activeScript.project_id, {
            target_lang: cfg.targetLang,
            source_lang: activeScript.source_lang,
          });
        } finally {
          unsubscribeSse();
          if (fallbackTimer !== undefined) {
            window.clearInterval(fallbackTimer);
          }
          await refreshParseStatus(true);
        }

        // 将解析出的真实剧本写回全局项目状态（仅当当前未切换项目时更新视图）
        if (isSameProject()) {
          useProjectStore.getState().setScript(activeScript);
        }
        setNodeStatus("media_input", "done");
        setNodeStatus("banquet", "done");
        setNodeStatus("whisperx", "done");
        setNodeStatus("llm_tagger", "done");
        setNodeStatus("script_hub", "done");

        const charCount = Object.keys(activeScript.characters).length;
        get().appendLog(
          "env_parse",
          `解析完成并已释放显存：生成 ${activeScript.segments.length} 条片段、${charCount} 个说话人`,
          "success",
        );

        if (parseOnly) {
          setActiveEdges([]);
          set({ isPipelineRunning: false });
          get().appendLog(
            "system",
            "仅解析模式已完成，可切回「剪辑」视图查看时间轴并按需合成配音",
            "success",
          );
          return;
        }
      }

      if (needsTranslate) {
        if (!await beforeNextStep("翻译已有对白")) return;
        activeNode = "llm_tagger";
        setNodeStatus("llm_tagger", "running");
        setActiveEdges(["e-llm-hub"]);
        get().appendLog("env_backend", `翻译已有 ${activeScript.segments.length} 条源台词，目标语言=${cfg.targetLang}；不会调用媒体解析。`, "info");
        const translated = await translateProject(targetProjectId, { target_lang: cfg.targetLang });
        activeScript = translated.script;
        if (isSameProject()) useProjectStore.getState().setScript(activeScript);
        for (const error of translated.errors ?? []) get().appendLog("env_backend", error, "warn");
        const partial = translated.skipped_count > 0 || Boolean(translated.errors?.length);
        setNodeStatus("llm_tagger", partial ? "error" : "done");
        setNodeStatus("script_hub", "done");
        get().appendLog("env_backend", `翻译已保存：成功 ${translated.translated_ids.length} 条，保留 ${translated.skipped_count} 条；时间轴已更新。`, partial ? "warn" : "success");
        if (partial) {
          get().appendLog("system", "部分台词翻译未完成，成功译文已保存；请检查日志后重试，未启动合成或导出。", "warn");
          return;
        }
      }
      if (parseOnly) {
        get().appendLog("system", "仅生成剧本已完成，未启动语音合成或成片导出。", "success");
        return;
      }

      if (activeScript.segments.length === 0) {
        setActiveEdges([]);
        set({ isPipelineRunning: false });
        get().appendLog("system", "未识别到可配音的语音片段，管线已停止", "warn");
        return;
      }

      // -------------------------------------------------------------------
      // 阶段 B：调用真实后端 POST /api/projects/{id}/synthesize（GPT-SoVITS）
      // -------------------------------------------------------------------
      if (needsSynthesize) {
        if (!await beforeNextStep("语音合成")) return;
        activeNode = "agent_tts";
        const synthesisBlocked = workflowPreflight(activeScript, { ...cfg, startStep: "synthesize" }, false);
        if (synthesisBlocked) throw new Error(synthesisBlocked);
        const pendingIds = activeScript.segments
          .filter((s) => !s.keep_original && (s.status !== "ready" || !s.tts_wav))
          .map((s) => s.id);

        setNodeStatus("agent_tts", "running");
        setActiveEdges(["e-hub-tts", "e-tts-mixer"]);

        if (pendingIds.length > 0) {
          get().appendLog(
            "env_tts",
            `开始调用 GPT-SoVITS 合成 ${pendingIds.length} 条待合成片段...`,
            "info",
          );
          const synthRes = await synthesizeSegments(activeScript.project_id, pendingIds);
          activeScript = synthRes.script;
          if (isSameProject()) {
            useProjectStore.getState().setScript(activeScript);
            useProjectStore.getState().bumpAudioVersion(synthRes.synthesized_ids);
          }

          for (const issue of synthesisIssues(synthRes)) {
            get().appendLog("env_tts", issue, "warn");
          }
          const incomplete = activeScript.segments.some((s) => !s.keep_original && (s.status !== "ready" || !s.tts_wav));
          setNodeStatus("agent_tts", incomplete ? "error" : "done");
          get().appendLog(
            "env_tts",
            `语音合成结束：成功 ${synthRes.synthesized_ids.length} 条`,
            incomplete ? "warn" : "success",
          );
        } else {
          get().appendLog("env_tts", "所有片段均为已就绪状态，跳过重复合成", "info");
          setNodeStatus("agent_tts", "done");
        }
      }

      if (!await beforeNextStep("成片导出")) return;
      const renderBlocked = renderPreflight(activeScript);
      if (renderBlocked) {
        activeNode = "ffmpeg_mixer";
        throw new Error(renderBlocked);
      }

      // -------------------------------------------------------------------
      // 阶段 C：调用真实后端 POST /api/projects/{id}/render（FFmpeg 混音封装）
      // -------------------------------------------------------------------
      activeNode = "ffmpeg_mixer";
      setNodeStatus("ffmpeg_mixer", "running");
      get().appendLog("env_backend", "正在执行毫秒拼轨与 FFmpeg 混音封装...", "info");

      const renderRes = await renderVideo(activeScript.project_id, { burn_watermark: cfg.burnWatermark, burn_subtitles: cfg.burnSubtitles });
      activeScript = {
        ...activeScript,
        media: { ...activeScript.media, output_video: renderRes.output_video },
      };
      if (isSameProject()) {
        useProjectStore.getState().setScript(activeScript);
      }
      setNodeStatus("ffmpeg_mixer", "done");
      get().appendLog("env_backend", `后续管线执行完毕，成片已输出：${renderRes.output_video}`, "success");

      setActiveEdges([]);
      set({ isPipelineRunning: false });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "管线执行异常";
      setNodeStatus(activeNode, "error");
      setActiveEdges([]);
      set({ isPipelineRunning: false });
      get().appendLog("system", `管线执行失败：${msg}`, "error");
      get().appendLog("system", "再次运行前请查看日志中的失败步骤。", "warn");
    } finally {
      releasePipelinePause = null;
      setActiveEdges([]);
      set((state) => ({
        isPipelineRunning: false,
        isPauseRequested: false,
        isPipelinePaused: false,
        isStopRequested: false,
        nodes: state.nodes.map((node) => {
          const status = node.data.status;
          if (status !== "queued" && status !== "running") return node;
          return { ...node, data: { ...node.data, status: status === "queued" ? "idle" : "error" } };
        }),
      }));
    }
  },

  runPipelineSimulation: async () => {
    if (get().isPipelineRunning) {
      return;
    }

    const setNodeStatus = (nodeId: string, status: NodeRunStatus) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, status } } : node,
        ),
      }));
    };

    const setActiveEdges = (activeEdgeIds: string[]) => {
      const activeSet = new Set(activeEdgeIds);
      set((state) => ({
        edges: state.edges.map((edge) => ({
          ...edge,
          animated: activeSet.has(edge.id),
        })),
      }));
    };

    set((state) => ({
      isPipelineRunning: true,
      nodes: state.nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: "queued" },
      })),
    }));
    get().appendLog("system", "开始执行全链路管线调度", "info");

    setNodeStatus("media_input", "running");
    setActiveEdges(["e-input-banquet"]);
    await sleep(350);
    setNodeStatus("media_input", "done");

    setNodeStatus("banquet", "running");
    setActiveEdges(["e-banquet-whisper", "e-banquet-hub-bgm"]);
    get().appendLog("env_parse", "Banquet 分离对白、歌声、伴奏和音效", "info");
    await sleep(500);
    setNodeStatus("banquet", "done");

    setNodeStatus("whisperx", "running");
    setActiveEdges(["e-whisper-llm"]);
    get().appendLog("env_parse", `MOSS BF16/SDPA 语音识别与窗口内匿名说话人分类`, "info");
    await sleep(600);
    setNodeStatus("whisperx", "done");

    setNodeStatus("llm_tagger", "running");
    setActiveEdges(["e-llm-hub"]);
    get().appendLog("env_parse", "口语化翻译与情绪动作标注", "info");
    await sleep(450);
    setNodeStatus("llm_tagger", "done");

    get().appendLog("env_parse", "解析子进程退出，显存已释放归零", "success");
    await sleep(350);

    setNodeStatus("script_hub", "running");
    setActiveEdges(["e-hub-tts", "e-hub-mixer-bgm"]);
    await sleep(350);
    setNodeStatus("script_hub", "done");

    setNodeStatus("agent_tts", "running");
    setActiveEdges(["e-tts-mixer"]);
    get().appendLog("env_tts", "按角色分组合成并检查时长约束 (4.2 GB)", "info");
    await sleep(700);
    setNodeStatus("agent_tts", "done");

    setNodeStatus("ffmpeg_mixer", "running");
    get().appendLog("env_backend", "混音封装完成", "success");
    await sleep(450);
    setNodeStatus("ffmpeg_mixer", "done");
    setActiveEdges([]);
    set({ isPipelineRunning: false });
  },
}));
