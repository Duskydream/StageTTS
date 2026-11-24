import createClient from "openapi-fetch";
import type { components, paths } from "../types/api.generated";
import type {
  AgentChatResponse,
  AutoEmotionRefsResponse,
  ExportDirsResponse,
  FinetuneJobStatus,
  FinetuneStartRequest,
  GpuDevice,
  GpuStatusResponse,
  HealthResponse,
  ParseStatusResponse,
  ProjectMode,
  ProjectScript,
  ReferencePreview,
  ReferenceUploadResponse,
  RenderRequest,
  RenderResponse,
  ScriptSnapshotListResponse,
  SubtitleLayout,
  SynthesizeResponse,
  TranslationResponse,
  UploadResponse,
  VoicePreset,
  VoicePresetCreateRequest,
  VoicePresetDeleteResponse,
  VoicePresetListResponse,
} from "../types/script";

export type { ExportDirsResponse, GpuDevice, HealthResponse, ParseStatusResponse, UploadResponse };

export class ApiError extends Error {
  status: number;
  isNotImplemented: boolean;
  isNetworkError: boolean;

  constructor(message: string, status: number, options?: { isNetworkError?: boolean }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.isNotImplemented = status === 501;
    this.isNetworkError = Boolean(options?.isNetworkError);
  }
}

export const apiClient = createClient<paths>({ baseUrl: "" });

async function unwrap<T>(
  promise: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  let result: { data?: T; error?: unknown; response: Response };
  try {
    result = await promise;
  } catch (err) {
    throw new ApiError(err instanceof Error ? err.message : "无法连接到 StageTTS 后端服务", 0, {
      isNetworkError: true,
    });
  }

  if (!result.response.ok || result.error !== undefined || result.data === undefined) {
    let detail = `请求失败 (${result.response.status})`;
    const errObj = result.error as { detail?: string } | undefined;
    if (errObj?.detail) {
      detail = errObj.detail;
    }
    throw new ApiError(detail, result.response.status);
  }

  return result.data;
}

export type TrainingSettingsResponse = components["schemas"]["TrainingSettingsResponse"];
export type TrainingSettings = components["schemas"]["TrainingSettings"];
export function fetchTrainingSettings(): Promise<TrainingSettingsResponse> {
  return unwrap(apiClient.GET("/api/settings/training", { cache: "no-store" }));
}
export function saveTrainingSettings(settings: TrainingSettings): Promise<TrainingSettingsResponse> {
  return unwrap(apiClient.PUT("/api/settings/training", { body: settings }));
}
export function applyInferenceBaseModel(
  baseModel: TrainingSettings["base_model"],
): Promise<TrainingSettingsResponse> {
  return unwrap(
    apiClient.PUT("/api/settings/training/inference", { body: { base_model: baseModel } }),
  );
}

export type LLMSettings = components["schemas"]["LLMSettingsResponse"];
export type LLMSettingsUpdate = components["schemas"]["LLMSettingsUpdate"];

export function fetchLLMSettings(): Promise<LLMSettings> {
  return unwrap(apiClient.GET("/api/settings/llm", { cache: "no-store" }));
}

export function saveLLMSettings(settings: LLMSettingsUpdate): Promise<LLMSettings> {
  return unwrap(apiClient.PUT("/api/settings/llm", { body: settings }));
}

export function checkHealth(): Promise<HealthResponse> {
  return unwrap(apiClient.GET("/api/health"));
}

export function fetchGpuStatus(signal?: AbortSignal): Promise<GpuStatusResponse> {
  return unwrap(apiClient.GET("/api/gpu", { signal, cache: "no-store" }));
}

export async function fetchProjects(): Promise<string[]> {
  const data = await unwrap(apiClient.GET("/api/projects"));
  return data.projects;
}

export function uploadProjectMedia(
  file: File,
  mode: ProjectMode = "video_dubbing",
  projectId = "",
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  if (projectId) {
    form.append("project_id", projectId);
  }
  return unwrap(
    apiClient.POST("/api/projects/upload", {
      body: form as unknown as components["schemas"]["Body_upload_project_media_api_projects_upload_post"],
    }),
  );
}

export function fetchProjectScript(projectId: string): Promise<ProjectScript> {
  return unwrap(
    apiClient.GET("/api/projects/{project_id}/script", {
      params: { path: { project_id: projectId } },
    }),
  );
}

export function saveProjectScript(script: ProjectScript): Promise<ProjectScript> {
  return unwrap(
    apiClient.PUT("/api/projects/{project_id}/script", {
      params: { path: { project_id: script.project_id } },
      body: script,
    }),
  );
}

export function uploadReference(projectId: string, file: File): Promise<ReferenceUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/references", {
      params: { path: { project_id: projectId } },
      body: form as unknown as components["schemas"]["Body_upload_project_reference_api_projects__project_id__references_post"],
    }),
  );
}

export function previewAutomaticReference(
  projectId: string,
  speaker: string,
): Promise<ReferencePreview> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/characters/{speaker}/reference-preview", {
      params: { path: { project_id: projectId, speaker } },
    }),
  );
}

export function autoMatchEmotionRefs(
  projectId: string,
  speaker: string,
  overwrite = false,
): Promise<AutoEmotionRefsResponse> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/characters/{speaker}/emotion-refs", {
      params: { path: { project_id: projectId, speaker } },
      body: { overwrite },
    }),
  );
}

export function startCharacterFinetune(
  projectId: string,
  speaker: string,
  payload: FinetuneStartRequest = {},
): Promise<FinetuneJobStatus> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/characters/{speaker}/finetune", {
      params: { path: { project_id: projectId, speaker } },
      body: payload,
    }),
  );
}

export function fetchFinetuneStatus(
  projectId: string,
  speaker?: string,
): Promise<FinetuneJobStatus> {
  return unwrap(
    apiClient.GET("/api/projects/{project_id}/finetune/status", {
      params: {
        path: { project_id: projectId },
        query: speaker ? { speaker } : undefined,
      },
      cache: "no-store",
    }),
  );
}

export function translateProject(
  projectId: string,
  options?: { target_lang?: string | null },
): Promise<TranslationResponse> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/translate", {
      params: { path: { project_id: projectId } },
      body: options,
    }),
  );
}

export function parseProjectMedia(
  projectId: string,
  options: { target_lang: string; source_lang?: string | null },
): Promise<ProjectScript> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/parse", {
      params: { path: { project_id: projectId } },
      body: {
        target_lang: options.target_lang,
        source_lang: options.source_lang && ["zh", "en", "ja", "ko", "yue"].includes(options.source_lang)
          ? options.source_lang as "zh" | "en" | "ja" | "ko" | "yue"
          : undefined,
      },
    }),
  );
}

export function fetchParseStatus(projectId: string): Promise<ParseStatusResponse> {
  return unwrap(
    apiClient.GET("/api/projects/{project_id}/parse/status", {
      params: { path: { project_id: projectId } },
    }),
  );
}

/**
 * 通过 SSE 实时订阅离线解析进度与日志，返回关闭订阅的清理函数。
 */
export function subscribeParseStatus(
  projectId: string,
  onUpdate: (status: ParseStatusResponse) => void,
  onError?: () => void,
): () => void {
  if (typeof EventSource === "undefined") {
    onError?.();
    return () => {};
  }
  const url = `/api/projects/${encodeURIComponent(projectId)}/parse/stream`;
  const source = new EventSource(url);
  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as ParseStatusResponse;
      onUpdate(data);
    } catch {
      // ignore malformed SSE chunk
    }
  };
  source.onerror = () => {
    source.close();
    onError?.();
  };
  return () => {
    source.close();
  };
}

export function synthesizeSegments(
  projectId: string,
  segmentIds?: number[],
): Promise<SynthesizeResponse> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/synthesize", {
      params: { path: { project_id: projectId } },
      body: { segment_ids: segmentIds ?? null },
    }),
  );
}

export function sendAgentChat(
  projectId: string,
  message: string,
  autoSynthesize = true,
): Promise<AgentChatResponse> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/agent/chat", {
      params: { path: { project_id: projectId } },
      body: { message, auto_synthesize: autoSynthesize },
    }),
  );
}

export function fetchExportDirs(): Promise<ExportDirsResponse> {
  return unwrap(apiClient.GET("/api/export/dirs"));
}

export function getSubtitlesUrl(projectId: string, layout: SubtitleLayout = "target"): string {
  return `/api/projects/${encodeURIComponent(projectId)}/subtitles?layout=${encodeURIComponent(layout)}`;
}

export function renderVideo(
  projectId: string,
  options: Partial<RenderRequest> = {},
): Promise<RenderResponse> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/render", {
      params: { path: { project_id: projectId } },
      body: {
        burn_watermark: false,
        burn_subtitles: "none",
        ...options,
      },
    }),
  );
}

export function fetchProjectHistory(projectId: string): Promise<ScriptSnapshotListResponse> {
  return unwrap(
    apiClient.GET("/api/projects/{project_id}/history", {
      params: { path: { project_id: projectId } },
    }),
  );
}

export function restoreProjectSnapshot(
  projectId: string,
  snapshotId: string,
): Promise<ProjectScript> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/history/{snapshot_id}/restore", {
      params: { path: { project_id: projectId, snapshot_id: snapshotId } },
    }),
  );
}

export function fetchVoicePresets(): Promise<VoicePresetListResponse> {
  return unwrap(apiClient.GET("/api/voice-presets"));
}

export function createVoicePreset(payload: VoicePresetCreateRequest): Promise<VoicePreset> {
  return unwrap(
    apiClient.POST("/api/voice-presets", {
      body: payload,
    }),
  );
}

export function deleteVoicePreset(presetId: string): Promise<VoicePresetDeleteResponse> {
  return unwrap(
    apiClient.DELETE("/api/voice-presets/{preset_id}", {
      params: { path: { preset_id: presetId } },
    }),
  );
}

export function analyzePerformance(
  projectId: string,
  segmentIds: number[],
  enableEmotion = false,
): Promise<components["schemas"]["PerformanceAnalysisResponse"]> {
  return unwrap(
    apiClient.POST("/api/projects/{project_id}/analyze-performance", {
      params: { path: { project_id: projectId } },
      body: { segment_ids: segmentIds, enable_emotion: enableEmotion, enable_prosody: true },
    }),
  );
}
