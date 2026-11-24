# StageTTS 全局架构与开发契约（ARCHITECTURE.md）

> 本文档是 **StageTTS** 全团队（成员 A/B/C/D）与 AI Coding 工具的**唯一架构与接口契约（Single Source of Truth）**。任何代码生成、模块重构或接口变更均须严格遵守本文档约定。

---

## 1. 项目定位与双模核心架构

**StageTTS** 是面向个人创作者、学习者与中小团队的本地化可控音视频生产平台（基准硬件：单卡消费级 GPU **NVIDIA RTX 5060 8GB**）。

系统以标准化剧本文件 **`workspace/{project_id}/script.json`** 作为全链路枢纽，彻底解耦上游“离线高显存感知解析/语音合成”与下游“轻量级交互精修/舞台渲染”，一套引擎原生支持两种工作模式：

- **模式 A（`video_dubbing`）**：多角色视频重配（外语网课、科普、动漫切片、短剧二创）。保留原画面与原 BGM，支持单句情绪切换、笑声/叹息原声保留、Agent 自然语言批处理及无损/合规水印导出。
- **模式 B（`virtual_stage`）**：单人叙事/播客虚拟演播室。离线合成高质量多情绪配音，前端基于 Web Audio API `AnalyserNode` 实时提取音频 RMS 能量驱动 Live2D（或 Three-VRM）嘴型，并随 `script.json` 时间节点触发表情与动作切换。

### 系统总体数据流

```text
[输入层] 原始视频 / 动漫切片 / 播客音频 / 脚本文本
   │
   ▼  (独立子进程 `env_parse`；退出后由 OS 回收显存)
[感知解析层] `workers/parse_worker.py`
   ├── Banquet 四轨分离  ──▶ 分离对白 / 歌声 / 伴奏 / 音效
   └── MOSS-Transcribe-Diarize ──▶ 联合 ASR、窗口内匿名角色与句段时间（无词级强制对齐）
   │
   ▼  输出全链路唯一真相源：workspace/{project_id}/script.json
[交互精修与 Agent 协同层] backend/ (env_backend, 纯 CPU) + frontend/ (Node.js)
   ├── GUI 手动精修：wavesurfer.js 波形色块轨道 + Inspector 单句修改与局部重合成
   └── Agent 协同闭环：大模型 Function Calling 调度 5 大确定性工具 + 时长反思改写闭环
   │
   ▼  (独立推理服务 env_tts，按 speaker 分组切换权重)
[离线合成与后期层]
   ├── GPT-SoVITS (api_v2.py, 127.0.0.1:9880) ──▶ 动态情绪参考音频路由 + 分组批合成
   └── pydub + FFmpeg (6.x+)                   ──▶ 原声保留 + 毫秒拼轨 + 混音导出
   │
   ├──▶ [模式 A：video_dubbing]   FFmpeg 混入原 BGM 与原画面流，输出 output.mp4
   └──▶ [模式 B：virtual_stage] 前端 Live2D / Three-VRM 读取 script.json + 音频实时演出
```

---

## 2. AI Coding 硬性铁律（Hard Invariants）

所有开发人员与 AI Coding 助手编写代码时，**严禁违背以下 9 条铁律**：

1. **四环境物理隔离，严禁混装依赖**
   - `env_parse`（独立 Python 3.12 / torch + torchaudio 2.8.0+cu128 / Transformers 5.6.0）：Banquet、固定版本 MOSS 联合转录。
   - `env_tts`（GPT-SoVITS CUDA 12.8 整合环境）：服务 `127.0.0.1:9880`。
   - `env_backend`（Python 3.10+、CPU 协调器）：FastAPI、Pydantic、LLM 客户端、文件/API 和任务调度。
   - `frontend`（Node.js 20 LTS + pnpm）：React、TypeScript、Vite。
   - WhisperX / pyannote 仅为 legacy worker 源码 / 实验，不列入 MOSS 依赖，不自动 fallback。

2. **RTX 5060 8GB（Blackwell `sm_120`）串行显存接力与单任务防 OOM**
   - `env_parse` 必须通过 `subprocess.run` 单独运行 `workers/parse_worker.py`，进程退出后由操作系统回收其 CUDA 显存。
   - MOSS 使用 BF16 / SDPA；在 RTX 5060 8GB 上已经测试最长 60 秒样例（峰值保留约 1.965 GiB），这不是 90 分钟资源保证。应用默认输入上限为 5400 秒（90 分钟），输出预算 65536 token；长视频分块、跨块 speaker linking 尚未支持。
   - 主后端必须使用 GPU 单任务锁，防止并发启动多个解析 / 合成任务。

3. **按说话人（`speaker`）分组批量合成**
   - 调用 GPT-SoVITS 合成多个片段时，必须先按 `segment.speaker` 分组，在切换到下一个角色前完成该角色名下全部待合成片段，**每个角色最多触发一次权重切换**，严禁按时间顺序逐句来回切换 `gpt_weights` / `sovits_weights`。

4. **动态情绪参考音频路由与零样本兜底**
   - 合成前根据 `segment.emotion` 查询 `characters[speaker].emotion_refs[emotion]`；若目标情绪不存在，自动回退到 `emotion_refs["neutral"]`，再无则取任意可用参考音。
   - 缺少可用情绪参考时，从本项目 `media.vocals_wav`（对白轨，兼容旧项目 `vocals.wav`）按同角色已对齐或人工确认的台词范围截取 3–10 秒，并使用对应原文。切换角色权重后恢复服务默认权重。

5. **Agent 严禁裸写 FFmpeg 或 Shell 命令**
   - 大模型 Agent 只能通过 OpenAI Function Calling 调度后端封装好的 **5 个确定性原子工具**（`query_script`、`batch_update_segments`、`synthesize_segments`、`update_mix_settings`、`render_final_video`）。禁止让 LLM 生成并执行任意 FFmpeg 命令或 Python 代码。

6. **时长约束反思闭环规范**
   - 每次合成单句完成后，工具层检测 `actual_duration` 与原时间槽 `slot_duration = end - start`：
     - 若 `actual_duration > slot_duration * 1.15`，**不得强行高倍加速**，必须将超标秒数与建议删减字数作为结构化反馈回传 LLM，触发简练口语化重写并重合成，**最多迭代 2 轮**（`MAX_REFLECTION_ROUNDS = 2`）。
     - 全部配音、全管线和智能体共用时长闭环。自动变速最多 1.15 倍，写入试听 WAV 后复测并上报仍超出时间槽的片段。手动语速在重配时生效。
     - 新 WAV 记录 `applied_speed_factor`，导出按成品直接拼轨。旧 WAV 缺少该字段时沿用导出时变速。

7. **音视频后期与 FFmpeg 硬水印坑点规避**
   - `pydub` 按毫秒 `overlay` 拼接完整人声轨（避免长命令行参数溢出）。当 `segment.keep_original == true` 时，直接从 `media.vocals_wav`（对白轨）对应 `[start, end]` 切片保留原声（笑声、叹息等），跳过 TTS 音频。
   - FFmpeg 混音必须使用 `amix=inputs=2:duration=first:normalize=0`（配合精确音量系数 `media.bgm_volume`），禁止使用默认 `normalize=1` 导致音量被莫名腰斩。
   - **关键坑点**：FFmpeg 的 `-c:v copy`（视频流直拷贝）**不能与 `-vf`（视频滤镜，如烧录右下角视觉水印）同时使用**，否则 FFmpeg 直接报错退出。因此：
     - **快速无损预览/导出（`burn_watermark=false`）**：使用 `-c:v copy`，合规标识由前端播放层或软字幕展示；
     - **合规硬水印成片导出（`burn_watermark=true`）**：必须去掉 `-c:v copy`，改用 `-c:v h264_nvenc`（RTX 5060 硬件加速，或降级 `libx264 -preset veryfast`）配合 `-vf drawtext=...` 烧录“AI 合成语音”水印。

8. **前端音频防缓存机制**
   - 单句重合成会覆盖同一个本地文件（如 `tts/seg_0000.wav`）。前端播放或加载波形/试听音频 URL 时，**必须拼接防缓存版本参数** `?t=<timestamp>`（由 Zustand `audioVersionMap` 维护），禁止直接缓存旧音频 URL。

9. **AI 伦理、声音版权与显式标识合规**
   - 测试与公开演示仅限使用**团队成员自录授权音色**或**开源无版权许可音色**，禁止未授权克隆商业声优、影视角色或公众人物声线。
   - 模式 A 视频播放器与模式 B 虚拟舞台 UI 默认显示“**AI 合成语音**”合规提示标识。

---

## 3. 核心中间表示：`script.json` 协议规范

`backend/schemas.py`（Pydantic v2）与 `frontend/src/types/script.ts`（TypeScript）必须与以下结构保持 **1:1 字段名与类型一致**。

### 3.1 完整协议示例

```json
{
  "project_id": "proj_demo_01",
  "mode": "video_dubbing",
  "media": {
    "source_video": "/workspace/proj_demo_01/input.mp4",
    "vocals_wav": "/workspace/proj_demo_01/cinematic-xxxx/dialogue.wav",
    "bgm_wav": "/workspace/proj_demo_01/cinematic-xxxx/bgm.wav",
    "cinematic_stems": {
      "dialogue": "/workspace/proj_demo_01/cinematic-xxxx/dialogue.wav",
      "singing": "/workspace/proj_demo_01/cinematic-xxxx/singing.wav",
      "music": "/workspace/proj_demo_01/cinematic-xxxx/music.wav",
      "effects": "/workspace/proj_demo_01/cinematic-xxxx/effects.wav"
    },
    "output_video": null,
    "duration": 45.2,
    "bgm_volume": 0.75
  },
  "characters": {
    "SPEAKER_00": {
      "display_name": "角色A",
      "color": "#3b82f6",
      "gpt_weights": "GPT_weights_v2/char_A.ckpt",
      "sovits_weights": "SoVITS_weights_v2/char_A.pth",
      "emotion_refs": {
        "neutral": { "wav": "/assets/refs/A_neutral.wav", "text": "今天天气不错。", "lang": "zh" },
        "angry":   { "wav": "/assets/refs/A_angry.wav",   "text": "你到底在干什么！", "lang": "zh" },
        "happy":   { "wav": "/assets/refs/A_happy.wav",   "text": "太好了，我们成功了！", "lang": "zh" },
        "sad":     { "wav": "/assets/refs/A_sad.wav",     "text": "原来是这样啊……", "lang": "zh" }
      },
      "avatar": { "type": "live2d", "model_url": "/assets/models/host_a/model3.json" }
    }
  },
  "segments": [
    {
      "id": 0,
      "start": 1.24,
      "end": 3.85,
      "speaker": "SPEAKER_00",
      "orig_text": "Wait, are you sure about this?",
      "target_text": "等等，你确定要这么做吗？",
      "target_lang": "zh",
      "emotion": "angry",
      "action": "shake_head",
      "keep_original": false,
      "speed_factor": 1.0,
      "tts_wav": "/workspace/proj_demo_01/tts/seg_0000.wav",
      "actual_duration": 2.41,
      "status": "ready"
    }
  ]
}
```

### 3.2 枚举与字段字典

- **`ProjectMode`**：`"video_dubbing"` | `"virtual_stage"`
- **`EmotionType`**：`"neutral"` | `"angry"` | `"happy"` | `"sad"` | `"whisper"`
- **`SegmentStatus`**：`"pending"` | `"synthesizing"` | `"ready"` | `"error"`
- **`AvatarType`**：`"live2d"` | `"vrm"`
- **`SubtitleBurnMode`**：`"none"` | `"target"` | `"bilingual"`
- **`SubtitleLayout`**：`"target"` | `"bilingual"`

| 层级 | 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| Root | `project_id` | `string` | 是 | 项目唯一 ID（对应 `workspace/{project_id}/` 目录名） |
| Root | `mode` | `ProjectMode` | 是 | 当前工作模式 |
| Root | `media` | `MediaInfo` | 是 | 媒体路径与全局混音参数 |
| Root | `source_lang`, `glossary` | `string \| null`, `Record<string, string>` | 否 | 检测到或手工设定的源语种、项目术语表 |
| Root | `characters` | `Record<string, CharacterConfig>` | 是 | 说话人映射字典（Key 如 `SPEAKER_00`） |
| Root | `segments` | `Segment[]` | 是 | 按时间排序的台词片段列表 |
| `media` | `source_video` | `string \| null` | 否 | 原视频/源音频 URL（如 `/workspace/{id}/input.mp4`） |
| `media` | `vocals_wav` | `string \| null` | 否 | 对白轨 URL |
| `media` | `bgm_wav` | `string \| null` | 否 | 歌声、伴奏和音效合成的背景轨 URL |
| `media` | `cinematic_stems` | `CinematicStems \| null` | 否 | 独立对白、歌声、伴奏、音效 WAV URL |
| `media` | `stem_volumes` | `StemVolumes \| null` | 否 | 四轨分离背景子轨独立音量系数（`singing` / `music` / `effects`，范围 `0.0` ~ `2.0`） |
| `media` | `output_video` | `string \| null` | 否 | 最终渲染输出 URL（如 `/workspace/{id}/output.mp4`） |
| `media` | `duration` | `number` | 是 | 项目总时长（秒） |
| `media` | `bgm_volume` | `number` | 是 | 伴奏混音音量主系数（`0.0` ~ `2.0`，默认 `0.75`） |
| `media` | `auto_ducking` | `boolean` | 否 | 人声出现时是否自动闪避背景音乐（默认 `true`） |
| `character` | `display_name` | `string` | 是 | 前端显示角色名 |
| `character` | `speaking_style` | `string` | 否 | 翻译与时长改写使用的角色语气 |
| `character` | `color` | `string` | 是 | 波形轨道色块 Hex 颜色（如 `#3b82f6`） |
| `character` | `gpt_weights` | `string \| null` | 否 | GPT-SoVITS GPT 模型权重路径 |
| `character` | `sovits_weights` | `string \| null` | 否 | GPT-SoVITS SoVITS 模型权重路径 |
| `character` | `emotion_refs` | `Partial<Record<EmotionType, EmotionRef>>` | 是 | 情绪参考音频库（`wav`, `text`, `lang`） |
| `character` | `avatar` | `AvatarConfig \| null` | 否 | 虚拟舞台形象配置（`type`, `model_url`） |
| `segment` | `id` | `number` | 是 | 片段整型唯一 ID（从 0 递增） |
| `segment` | `start` | `number` | 是 | 起始时间（秒） |
| `segment` | `end` | `number` | 是 | 结束时间（秒，必须 `> start`） |
| `segment` | `speaker` | `string` | 是 | 关联 `characters` 的 Key |
| `segment` | `orig_text` | `string` | 是 | 原语种识别文本 |
| `segment` | `words` | `AlignedWord[]` | 否 | 词及绝对时间范围、原聚类标签，缺失值为空 |
| `segment` | `alignment_needs_review`, `speaker_needs_review` | `boolean` | 否 | 原文范围和说话人校对状态，旧剧本默认待校对 |
| `segment` | `target_text` | `string` | 是 | 目标配音文本 |
| `segment` | `target_lang` | `string` | 是 | 目标语种代码（如 `zh`, `en`, `ja`） |
| `segment` | `emotion` | `EmotionType` | 是 | 情绪标签 |
| `segment` | `action` | `string \| null` | 否 | 虚拟舞台预制动作标签（如 `shake_head`, `nod`） |
| `segment` | `keep_original` | `boolean` | 是 | `true` 时保留 `media.vocals_wav` 原声，不做 TTS 替换 |
| `segment` | `speed_factor` | `number` | 是 | 温和语速调整系数（默认 `1.0`，建议上限 `1.15`） |
| `segment` | `tts_wav` | `string \| null` | 否 | 单句合成音频 URL（如 `/workspace/{id}/tts/seg_0000.wav`） |
| `segment` | `applied_speed_factor` | `number \| null` | 否 | 已写入 WAV 的倍率，旧项目为空 |
| `segment` | `actual_duration` | `number \| null` | 否 | 实际合成音频时长（秒） |
| `segment` | `status` | `SegmentStatus` | 是 | 片段状态 |

---

## 4. REST API 与静态资源接口契约

后端运行于 `http://127.0.0.1:8000`；前端 Vite 开发服务器将 `/api`、`/workspace`、`/assets` 代理至后端。

| 方法 | 路径 | 说明 | 请求体 / 参数 | 响应体 |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | 健康检查（含目录与任务锁状态） | — | `{ "status": "ok", "busy": false }` |
| `GET` | `/api/projects` | 列出 `workspace/` 下所有含 `script.json` 的项目 | — | `{ "projects": ["proj_demo_01", ...] }` |
| `POST` | `/api/projects/upload` | 创建项目并上传源视频/音频 | `multipart/form-data` (`file`, `mode?`, `project_id?`) | `{ "project_id": "...", "media_url": "..." }` |
| `POST` | `/api/projects/{project_id}/parse` | 触发 `env_parse` 子进程解析并生成 `script.json`（受单任务锁保护） | `ParseRequest`（含目标语种和识别精度） | `ProjectScript` |
| `GET` | `/api/projects/{project_id}/script` | 读取指定项目的 `script.json` | — | `ProjectScript` |
| `PUT` | `/api/projects/{project_id}/script` | 保存前端修改后的完整 `ProjectScript`（Pydantic 校验后落盘，并自动保存历史快照） | `ProjectScript` | `ProjectScript` |
| `GET` | `/api/projects/{project_id}/history` | 列出项目最近的历史快照列表（最多 20 条，倒序） | — | `ScriptSnapshotListResponse` |
| `POST` | `/api/projects/{project_id}/history/{snapshot_id}/restore` | 将项目回滚恢复至指定历史快照（同时保留当前版本快照） | — | `ProjectScript` |
| `GET` | `/api/voice-presets` | 列出全局跨项目音色预设库 | — | `VoicePresetListResponse` |
| `POST` | `/api/voice-presets` | 将当前角色音色（权重与情绪参考音频）另存为跨项目预设并复制参考音频 | `VoicePresetCreateRequest` | `VoicePreset` |
| `DELETE` | `/api/voice-presets/{preset_id}` | 删除指定全局音色预设并清理其隔离参考音频 | — | `VoicePresetDeleteResponse` |
| `POST` | `/api/projects/{project_id}/references` | 上传 3–10 秒音频并转换为 WAV | `multipart/form-data` (`file`) | `{ wav }` |
| `POST` | `/api/projects/{project_id}/characters/{speaker}/reference-preview` | 从同角色已校对原声生成试听 | — | `ReferencePreview` |
| `POST` | `/api/projects/{project_id}/translate` | 按当前术语与角色设置重译，保留失败的台词 | — | `TranslationResponse` |
| `POST` | `/api/projects/{project_id}/synthesize` | 按 `speaker` 分组批量合成指定片段并返回时长质检结果 | `SynthesizeRequest` | `SynthesizeResponse` |
| `POST` | `/api/projects/{project_id}/agent/chat` | 自然语言批量剪辑对话（多轮工具调用 + 超时反思改写） | `AgentChatRequest` | `AgentChatResponse` |
| `GET` | `/api/projects/{project_id}/subtitles` | 导出当前剧本的 ASS 字幕文件（`layout=target\|bilingual`） | Query: `layout` | UTF-8-SIG ASS 文本 |
| `GET` | `/api/export/dirs` | 获取工作区与本机推荐导出目录（视频/下载/桌面） | — | `ExportDirsResponse` |
| `POST` | `/api/projects/{project_id}/render` | `pydub` 拼轨 + 四轨独立混音/自动闪避 + FFmpeg 导出成片，支持自定义文件名/本机目录落盘、ASS 硬字幕与外挂 ASS 字幕 | `RenderRequest` | `RenderResponse` |
| `STATIC` | `/workspace/*` | 静态挂载 `workspace/` 目录（提供视频、人声、伴奏、单句 TTS 音频） | — | 二进制流 |
| `STATIC` | `/assets/*` | 静态挂载 `assets/` 目录（提供 Live2D/VRM 模型与情绪参考音频） | — | 二进制流 |

---

## 5. Agent 五大确定性原子工具契约

定义于 `backend/agent/tools.py`，供 `backend/agent/orchestrator.py` 在 Function Calling 循环中调度：

| 工具名 | 参数签名 | 返回值结构 | 副作用 |
| :--- | :--- | :--- | :--- |
| `query_script` | `project_id: str`, `start_sec: float \| None = None`, `end_sec: float \| None = None`, `speaker: str \| None = None` | `{ "count": int, "segments": list[dict] }` | 只读查询 `script.json`，不写盘 |
| `batch_update_segments` | `project_id: str`, `updates: list[SegmentPatch]`（含 `id` 及可选 `start` / `end` / `speaker` / `orig_text` / `target_text` / `emotion` / `action` / `keep_original` / `speed_factor`） | `{ "updated_ids": list[int], "marked_pending_ids": list[int] }` | 修改指定片段字段；若修改了 `start`/`end`/`speaker`/`target_text`/`target_lang`/`emotion`/`keep_original`/`speed_factor`，自动将 `status` 置为 `"pending"` 并持久化 `script.json`（修改前保存历史快照） |
| `synthesize_segments` | `project_id: str`, `segment_ids: list[int] \| None = None`（为 `None` 时合成全部 `status != "ready"` 且 `keep_original == false` 的片段） | `{ "synthesized_ids": list[int], "overflow_segments": list[SegmentDurationCheck], "errors": list[str] }` | 按角色分组配音，执行时长闭环并返回最终剩余超长 |
| `update_mix_settings` | `project_id: str`, `bgm_volume: float \| None = None`, `singing_volume: float \| None = None`, `music_volume: float \| None = None`, `effects_volume: float \| None = None`, `auto_ducking: bool \| None = None` | `{ "bgm_volume": float, "stem_volumes": StemVolumes \| null, "auto_ducking": bool }` | 更新背景混音主音量、四轨子音量与自动闪避开关并持久化 `script.json` |
| `render_final_video` | `project_id: str`, `burn_watermark: bool = False`, `burn_subtitles: SubtitleBurnMode = "none"`, `auto_ducking: bool \| None = None` | `{ "output_video": str, "duration": float, "burn_watermark": bool, "burn_subtitles": SubtitleBurnMode, "auto_ducking": bool }` | 触发 `pydub` 拼轨与 FFmpeg 混音封装（支持 ASS 硬字幕烧录与自动闪避），写回 `media.output_video` 并持久化 `script.json` |

---

## 6. 4 人并行协作边界与契约修改流程

| 成员 | 目录与文件所有权 | 核心职责 |
| :--- | :--- | :--- |
| **成员 A** | `workers/parse_worker.py`, `workers/moss_worker.py`, `workers/aligned_segments.py`, `experiments/` | 维护固定 revision 的 MOSS / Banquet 主解析、MOSS 证据与离线评测；WhisperX / pyannote 只作为显式 legacy 研究代码 |
| **成员 B** | `backend/`（`main.py`, `config.py`, `schemas.py`, `agent/`, `services/`） | 维护 `env_backend` 与 `env_tts` 联调、GPT-SoVITS 分组合成、Agent 5 工具与反思闭环、FFmpeg 混音 |
| **成员 C** | `frontend/`（除 `VirtualStage.tsx` 外的全量组件、Store 与样式） | 类 PR 三栏工作台、`wavesurfer.js` 波形色块轨道、`Inspector` 单句精修、`AgentChat` 对话交互 |
| **成员 D** | `frontend/src/components/VirtualStage.tsx`、`assets/`、`README.md` | Live2D/Three-VRM 虚拟演播室嘴型与动作驱动、授权情绪参考音库、合规提示与演示交付 |

**契约变更流程**：任何涉及 `script.json` 结构、REST API 路径/字段或 Agent 工具签名的改动，必须先由提出方同步更新本文件（`ARCHITECTURE.md`）、`backend/schemas.py` 与 `frontend/src/types/script.ts` 并在群内通知全员对齐后方可合并代码。

---

## 7. 技术栈锁定版本参考

| 层级 | 锁定技术栈与版本范围 |
| :--- | :--- |
| **前端核心** | Node.js 20 LTS, `pnpm`, Vite `^5.4.0`, React `^18.3.1`, TypeScript `^5.5.0`, TailwindCSS `^4.0.0`, Zustand `^4.5.0`, `@tanstack/react-query` `^5.62.0`, `openapi-fetch` `^0.13.4`, `@xyflow/react` `^12.3.6`, `lucide-react` `^0.469.0`, `@biomejs/biome` `^1.9.4` |
| **波形与舞台** | `wavesurfer.js` `^7.8.0` (含 `@wavesurfer/react` 或 `regions` 插件), `pixi.js` `^7.4.0`, `pixi-live2d-display-lipsyncpatch` `^0.5.0-ls-8`（可选进阶 `three` + `@pixiv/three-vrm` v3） |
| **主后端** | Python 3.10+, `fastapi>=0.136.0`, `starlette>=1.3.1`, `uvicorn[standard]>=0.30.0`, `pydantic>=2.7.0`, `pydantic-settings>=2.4.0`, `openai>=3.57.0`, `httpx>=0.27.0`, `pydub>=0.25.1`, `PyYAML>=6.0.1` |
| **感知解析** | 独立 Python 3.12、PyTorch / torchaudio 2.8.0+cu128、Transformers 5.6.0、BF16 / SDPA、Banquet 与 pin revision 的 MOSS-Transcribe-Diarize |
| **语音合成** | GPT-SoVITS (`api_v2.py`, CUDA 12.8 适配版, `is_half: true`) |
| **音视频工具** | FFmpeg `6.x+` (含 `nvenc` / `libx264` 与 `amix` 滤镜) |
