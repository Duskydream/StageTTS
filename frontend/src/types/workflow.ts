/**
 * ComfyUI 节点流管线与 Premiere 工作区相关类型定义。
 */

import type { SubtitleBurnMode } from "./script";

export type WorkspacePreset = "editing" | "workflow";

export type CenterViewTab = "video" | "graph";

export type RightPanelTab = "inspector" | "agent" | "logs";

export type LeftDrawerTab = "none" | "characters" | "media";

export type EnvBadge = "env_parse" | "env_tts" | "env_backend";

export type NodeCategory = "input" | "parse" | "hub" | "synthesis" | "output";

export type NodeRunStatus = "idle" | "queued" | "running" | "done" | "error";

export interface PortSpec {
  id: string;
  label: string;
  type: string;
}

export interface StageNodeData extends Record<string, unknown> {
  title: string;
  category: NodeCategory;
  env: EnvBadge;
  vramGb: number | null;
  status: NodeRunStatus;
  inputs: PortSpec[];
  outputs: PortSpec[];
  summary: string;
  params: Record<string, string | number | boolean>;
}

export type PipelineStartStep = "parse" | "translate" | "synthesize" | "render";

export function pipelineStartForNode(nodeId: string | null): PipelineStartStep | null {
  switch (nodeId) {
    case "media_input":
    case "banquet": return "parse";
    case "llm_tagger": return "translate";
    case "script_hub":
    case "agent_tts": return "synthesize";
    case "ffmpeg_mixer": return "render";
    default: return null;
  }
}

export function pipelineNodeHint(nodeId: string | null): string {
  if (nodeId === "whisperx") return "从音轨分离或翻译节点启动";
  return "点击节点选择起点";
}

export interface PipelineConfig {
  targetLang: "zh" | "en" | "ja";
  burnWatermark: boolean;
  burnSubtitles: SubtitleBurnMode;
  naturalPolish: boolean;
  webSearch: boolean;
}

export interface PipelineLogEntry {
  id: string;
  timestamp: string;
  env: EnvBadge | "system";
  message: string;
  level: "info" | "success" | "warn" | "error";
}
