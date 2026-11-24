/**
 * StageTTS 核心 TypeScript 数据契约。
 * 由 OpenAPI（backend/schemas.py -> api.generated.ts）自动生成并统一导出，避免手写双份类型漂移。
 */

import type { components } from "./api.generated";

export type CinematicStems = components["schemas"]["CinematicStems"];
export type StemVolumes = components["schemas"]["StemVolumes"];
export type MediaInfo = components["schemas"]["MediaInfo"];
export type EmotionRef = components["schemas"]["EmotionRef"];
export type AvatarConfig = components["schemas"]["AvatarConfig"];
export type CharacterConfig = components["schemas"]["CharacterConfig"];
export type AlignedWord = components["schemas"]["AlignedWord"];
export type Segment = components["schemas"]["Segment"];
export type ProjectScript = components["schemas"]["ProjectScript-Output"];

export type ProjectMode = ProjectScript["mode"];
export type EmotionType = Segment["emotion"];
export type SegmentStatus = Segment["status"];
export type AvatarType = AvatarConfig["type"];
export type SubtitleBurnMode = components["schemas"]["RenderRequest"]["burn_subtitles"];
export type SubtitleLayout = Exclude<SubtitleBurnMode, "none">;

export type SegmentPatch = { id: Segment["id"] } & Partial<
  Pick<
    Segment,
    | "start"
    | "end"
    | "speaker"
    | "orig_text"
    | "alignment_needs_review"
    | "segment_timing_needs_review"
    | "speaker_needs_review"
    | "target_text"
    | "target_lang"
    | "emotion"
    | "action"
    | "keep_original"
    | "speed_factor"
  >
>;

export type ReadinessInfo = components["schemas"]["ReadinessInfo"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type GpuDevice = components["schemas"]["GpuDevice"];
export type GpuStatusResponse = components["schemas"]["GpuStatusResponse"];
export type ProjectListResponse = components["schemas"]["ProjectListResponse"];
export type ReferenceUploadResponse = components["schemas"]["ReferenceUploadResponse"];
export type UploadResponse = components["schemas"]["UploadResponse"];
export type ParseStatusResponse = components["schemas"]["ParseStatusResponse"];
export type ParseRequest = components["schemas"]["ParseRequest"];
export type ReferencePreview = components["schemas"]["ReferencePreview"];
export type TranslationResponse = components["schemas"]["TranslationResponse"];
export type SegmentDurationCheck = components["schemas"]["SegmentDurationCheck"];
export type SynthesizeResponse = components["schemas"]["SynthesizeResponse"];
export type ToolCallLog = components["schemas"]["ToolCallLog"];
export type AgentChatResponse = components["schemas"]["AgentChatResponse"];
export type ExportDirOption = components["schemas"]["ExportDirOption"];
export type ExportDirsResponse = components["schemas"]["ExportDirsResponse"];
export type RenderRequest = components["schemas"]["RenderRequest"];
export type RenderResponse = components["schemas"]["RenderResponse"];
export type ScriptSnapshotMeta = components["schemas"]["ScriptSnapshotMeta"];
export type ScriptSnapshotListResponse = components["schemas"]["ScriptSnapshotListResponse"];
export type VoicePreset = components["schemas"]["VoicePreset"];
export type VoicePresetCreateRequest = components["schemas"]["VoicePresetCreateRequest"];
export type VoicePresetListResponse = components["schemas"]["VoicePresetListResponse"];
export type VoicePresetDeleteResponse = components["schemas"]["VoicePresetDeleteResponse"];
export type AutoEmotionRefsResponse = components["schemas"]["AutoEmotionRefsResponse"];
export type FinetuneStartRequest = components["schemas"]["FinetuneStartRequest"];
export type FinetuneJobStatus = components["schemas"]["FinetuneJobStatus"];
