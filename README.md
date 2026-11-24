# StageTTS — 多角色视频配音与虚拟演播平台

> 《人工智能专业导论》课程落地应用项目。  
> 本项目把“视频理解 / 剧本生成 / 智能剪辑 / 多角色配音 / 成片导出”串成一条本地可控的流水线，核心中间格式是 `script.json`。

---

## 0. 第一次接触请先读这里

如果你刚加入项目，建议按这个顺序理解：

1. **这个项目解决什么问题？**  
   给一段视频或音频，自动分离四轨（对白/歌声/音乐/音效）、联合识别原文与说话人、经剧本分析与口语润色翻译成目标语言、标注情绪和动作，再按需选择 **GPT-SoVITS（支持角色微调）** 或 **CosyVoice 3（零样本）** 生成多角色配音，最后用 FFmpeg 混音并烧录字幕导出成片。

2. **项目的核心文件是什么？**  
   是 `workspace/{project_id}/script.json`。  
   它不是普通配置文件，而是整个系统的“唯一真相源”：
   - 解析结果写进去；
   - 前端时间轴读它；
   - 用户改台词也改它；
   - GPT-SoVITS / CosyVoice 3 根据它合成音频；
   - 成片导出也根据它拼轨并生成字幕。

3. **如果想先看界面效果，最少需要什么？**  
   只需要：
   - Node.js 20+
   - Python 3.10+
   - FFmpeg

   可以只启动后端 + 前端浏览工程与编辑界面。  
   如果要真正跑解析和配音，还要额外配置独立的 `env_parse`（Banquet + MOSS）、以及 GPT-SoVITS 或 CosyVoice 3 环境。

4. **不同电脑不用完全照抄开发机路径**  
   本 README 中的 `F:\Download\StageTTS`、`C:\Project\GPT-SoVITS...` 只是当前开发机示例。将 `ENV_PARSE_PYTHON` 指向本机独立 MOSS 环境，并修改 `MOSS_MODEL_PATH`、Banquet、GPT-SoVITS 和缓存路径即可。

---

> 当前配置更新：MOSS 输入上限已调整为 90 分钟（5400 秒），输出预算 65536 token，输入上下文 131072 token。以下历史 60 秒实测是资源测量范围，不再是应用上限。长音频的显存、耗时与质量需实际验证。

## 1. 项目一句话架构

```text
视频/音频
  ↓
Stage0 FFmpeg 抽取音频
  ↓
Stage1 Banquet 分离对白、歌声、伴奏和音效
  ↓
Stage2 MOSS-Transcribe-Diarize 联合 ASR + 匿名说话人 + 句段时间
  ↓
Stage3 对齐候选、证据保留、人工或一键校对（MOSS 不提供词级强制对齐）
  ↓
Stage4 LLM 三步配音翻译（全片分析 → 可选联网专名校对 → 初译与口语润色）
  ↓
script.json
  ↓
GPT-SoVITS（端口 9880）或 CosyVoice 3 零样本（端口 9881）按句合成配音
  ↓
FFmpeg 混音与可选 ASS 硬字幕烧录导出 output.mp4
```

MOSS 主模型固定至本地已审核 revision `10304d926740cf5a5cee05a268086e3265197910`，通过独立 `env_parse` 子进程运行；Banquet 分离、MOSS 联合转录与 GPT-SoVITS 合成串行接力显存。

**时长边界：**默认输入上限为 90 分钟（5400 秒），输出预算 65536 token，处理器上下文为 131072 token。模型按实际剩余上下文分配输出空间。当前 8GB 显卡实测仍只覆盖至 60 秒；长音频可能受到显存、耗时、上下文或输出长度限制，不能将放宽上限视为 90 分钟实测保证。

MOSS 输出的是模型联合生成的**句段时间**与窗口内匿名角色标记；不是词级时间、独立重叠检测或跨窗口全局角色身份。保持来源证据和校对提示，禁止将未知角色静默映射为已确认说话人。

---

## 2. 当前开发进度与已知边界

请组员特别注意：**“代码已有”不等于“主入口已经完整接通”。**

| 模块 | 当前状态 | 说明 |
| :--- | :---: | :--- |
| MOSS 联合 ASR 与 diarization | 主解析路径 | Banquet → MOSS → LLM。已完成 MOSS 10 / 30 / 60 秒英日真实推理及独立短样例全流程接入。当前输入上限为 90 分钟，实测覆盖至 60 秒；90 分钟是上游模型声明，不是本应用验证或承诺。MOSS 仅输出句段时间，词级对齐和准确率仍需审核。|
| WhisperX + pyannote | 仅历史 / 显式回滚 | 不是默认或静默 fallback；缺少 MOSS 配置或推理失败时明确报错。|
| `script.json` 数据契约 | 已完成 | Pydantic v2 与 TypeScript 类型 1:1 对齐 |
| 三步自然翻译与联网校对 | 已完成 | Pass 0 全片设定与术语抽取 → 可选 Pass 0.5 免费联网专名校对（Wikipedia + DuckDuckGo）→ Pass 1 初译 → Pass 2 配音口语润色，ComfyUI 节点卡内可直接开关 |
| 双 TTS 引擎切换 | 已完成 | 支持 **GPT-SoVITS**（端口 `9880`，支持 `v2 / v2Pro / v2ProPlus` 角色微调训练与音色预设库）与 **CosyVoice 3 零样本**（端口 `9881`，8GB 显卡 fp16 低显存加载，自动同语种 `zero_shot` / 跨语种 `cross_lingual`）并在设置页一键切换 |
| 8GB 显存串行接力 | 已完成 | 解析前释放正在运行的 TTS 服务，配音时按所选引擎按需拉起 |
| 配音时长闭环与导出贴合 | 已完成 | 双向时长反思改写（>1.15x 精简 / <0.8x 补足）+ `0.85–1.15x` 温和调速；`/render` 导出默认开启 `auto_fit_overflow` 对轻微侵入下一句的片段自动温和 `atempo` 贴合（最高 `1.35x`） |
| 字幕烧录与工作区交互 | 已完成 | 支持 Premiere 风格时间轴拖拽平移/边缘拉伸、角色面板与时间轴一键确认待校对、ComfyUI 节点参数内嵌及 FFmpeg 单次烧录 ASS 硬字幕（`无 / 原文（识别） / 译文 / 双语`） |
| 参考音与角色校对 | 已完成 | 在左侧「角色面板」编辑/合并角色、一键微调、上传或试听参考音与预设；自动参考音从通过校对、无说话人重叠的同角色原句中挑选 |
| 评测脚本 | 骨架就绪 | 需要接入真实评测集后才能真正跑出指标 |

---

## 3. 核心概念：script.json 是什么？

### 3.1 它存什么？

`script.json` 顶层结构如下：

```json
{
  "project_id": "proj_xxx",
  "mode": "video_dubbing",
  "media": {
    "source_video": "/workspace/proj_xxx/input.mp4",
    "vocals_wav": "/workspace/proj_xxx/cinematic-xxxx/dialogue.wav",
    "bgm_wav": "/workspace/proj_xxx/cinematic-xxxx/bgm.wav",
    "cinematic_stems": {
      "dialogue": "/workspace/proj_xxx/cinematic-xxxx/dialogue.wav",
      "singing": "/workspace/proj_xxx/cinematic-xxxx/singing.wav",
      "music": "/workspace/proj_xxx/cinematic-xxxx/music.wav",
      "effects": "/workspace/proj_xxx/cinematic-xxxx/effects.wav"
    },
    "output_video": null,
    "duration": 60.0,
    "bgm_volume": 0.75
  },
  "characters": {
    "SPEAKER_00": {
      "display_name": "角色 A",
      "color": "#3b82f6",
      "gpt_weights": null,
      "sovits_weights": null,
      "emotion_refs": {},
      "avatar": {
        "type": "live2d",
        "model_url": "/assets/models/host_a/model3.json"
      }
    }
  },
  "segments": [
    {
      "id": 0,
      "start": 2.8,
      "end": 10.5,
      "speaker": "SPEAKER_00",
      "orig_text": "原片识别文本",
      "target_text": "目标配音文本",
      "target_lang": "zh",
      "emotion": "neutral",
      "action": "idle",
      "keep_original": false,
      "speed_factor": 1.0,
      "tts_wav": null,
      "actual_duration": null,
      "status": "pending"
    }
  ]
}
```

### 3.2 Stage4 翻译怎么生成？

Stage4 使用**全片分析 + 全文上下文分批初译 + 口语化润色**（每批最多 25 条输出）：

1. **Pass 0 全片设定分析**（开启「口语润色」或「联网校对」时）：通读完整源对白，归纳题材基调、角色说话风格（非破坏式补全空 `speaking_style`）与核心术语；
2. **Pass 0.5 联网专名校对**（在 ComfyUI「翻译与情绪」节点勾选「联网校对」时）：仅对最多 8 个不确定专名/IP 译名执行免 Key 检索（Wikipedia + DuckDuckGo），校对固定译法但不覆盖人工术语；
3. **Pass 1 忠实初译**：携带完整源对白、`show_bible`、声学线索和软字数预算逐条输出译文、情绪、动作及原声事件建议，缺失/疑漏触发单句修复；
4. **Pass 2 配音口语润色**（开启「口语润色」时）：按角色口吻、气口断句去翻译腔改写，并通过术语/数字/缩写保真校验，未通过则回退 Pass 1 初译。

正常短应答（Yes、Go、你好、はい等）必须翻译，不能仅因短或 LLM 返回布尔值就保留外语原声；自动保留仅允许明确非语言事件记号。

### 3.3 它给 GPT-SoVITS 提供什么？

真正调用 GPT-SoVITS 时，每句会组装成：

```json
{
  "text": "segment.target_text",
  "text_lang": "segment.target_lang",
  "ref_audio_path": "参考音频路径",
  "prompt_text": "参考音频对应文本",
  "prompt_lang": "参考音频语种",
  "speed_factor": 1.0,
  "media_type": "wav"
}
```

其中：

| GPT-SoVITS 参数 | 来源 |
| :--- | :--- |
| `text` | `segments[i].target_text` |
| `text_lang` | `segments[i].target_lang` |
| `speed_factor` | `/tts` 固定传 `1.0`；片段 `speed_factor` 在合成后由 FFmpeg `atempo` 应用 |
| `ref_audio_path` | `characters[speaker].emotion_refs[emotion].wav`，没有则从 `media.vocals_wav` 生成自动候选 |
| `prompt_text` | 情绪参考音文本，或自动候选对应的 `orig_text` |
| `prompt_lang` | 情绪参考音语种，或自动候选使用 `script.source_lang` |

当前自动参考音逻辑如下：

1. `characters[speaker].emotion_refs[emotion]` 中人工上传/绑定的参考音优先级最高；
2. 没有人工参考音时，按 `segment.emotion → neutral → 任意可用情绪` 回退；
3. 仍然没有时，才调用 `automatic_reference()` 从 `media.vocals_wav` 指向的音轨 生成自动候选；
4. 已校对路径选择**原文已校对、说话人已确认、3–10 秒、没有与其他说话人重叠**的同角色连续原句；MOSS 未校对句段使用第 6 条的独立重识别路径；
5. 优先使用完整 3–10 秒已确认台词；长段只有在词级文本严格对应当前 `orig_text`、时间有效、词级说话人明确且没有其他角色重叠时才能按词边界裁剪，参考文本只包含裁剪窗口中的词；
6. MOSS 句段级结果没有合适的已校对参考时，自动从同角色、无已知重叠的句段选取最多 8 秒音频，在独立 CPU 子进程中用本地 MOSS 重新识别这段音频；识别完整、单说话人才使用裁剪音频对应的新逐字稿。首次较慢，音频及带来源信息的逐字稿会缓存在项目 references 目录；人工审核状态不变，角色归属仍继承模型预测。UNKNOWN 不自动取用，失败时提示上传参考音；参考语言始终填音频真实语种；
7. GPT-SoVITS 的 `/tts` 请求里 `speed_factor` 固定传 `1.0`，角色/片段的 `speed_factor` 是在后端用 FFmpeg `atempo` 对成品 WAV 做温和变速，试听和导出使用同一份处理后的文件。

### 3.4 M0 正确性边界

- 解析跳过/失败的说话人标为 `UNKNOWN`，保留失败原因并待校对，不冒充已经确认的单人；在精修面板指定真实角色后再确认。
- 长段优先自然停顿与句法边界；中日逐字对齐不能用任意 14 秒边界强切。允许受控延长，仍无可靠边界时保留完整原文并标记待审。
- 导出保留完整配音 WAV。超出原句槽但能放进后续空隙的尾部可保留；若新增串话或超过项目/源视频边界则明确阻塞导出待审，不删波形尾部、不改变源视频时长。旧项目的导出调速同样在调速后校验。
- M0 没有引入新说话人模型、声学情绪模型或新的 TTS。现有五类情绪只用于文本标记/参考音路由，不代表 GPT-SoVITS 接受任意情绪指令，也不保证源片表演或跨语种韵律逐词复现。

---

## 4. 环境准备总览

### 4.1 你需要准备哪些东西？

| 环境 | 作用 | 是否必须 | 推荐版本 |
| :--- | :--- | :---: | :--- |
| FFmpeg | 抽音频、探测时长、最终混音 | 必须 | 6.x+，开发机为 9.1 |
| Node.js | 前端 Vite 项目 | 必须 | Node.js 20 LTS |
| Python backend | 跑 FastAPI 后端 | 必须 | Python 3.10+ |
| GPT-SoVITS | 生成配音（支持微调） | 至少配置一种 TTS 引擎 | RTX 50 系列请用 CUDA 12.8 适配版 |
| CosyVoice 3 | 零样本配音（端口 `9881`） | 可选，与 GPT-SoVITS 二选一 | 独立 Python 3.10 + `Fun-CosyVoice3-0.5B-2512`，8GB 显卡 fp16 约 2.0 GB 显存 |
| env_parse | Banquet 四轨分离 + MOSS 联合识别/说话人/句段时间 | 完整解析必须 | 独立 Python 3.12 + torch / torchaudio 2.8.0+cu128 + Transformers 5.6.0 |
| Hugging Face Token | 当前本地 MOSS 主解析 | 不需要 | pinned 本地 snapshot；推理时不下载 |
| WhisperX / pyannote | 明确选择 legacy 时的历史路径 | 默认不需要 | 不参与 MOSS 主依赖或自动回退 |
| LLM 网关 | 翻译、情绪、动作标注、Agent | 强烈建议 | OpenAI-compatible API 即可 |

---

## 4.2 按电脑情况选择配置

### 情况 A：只想看界面 / 改前端

你需要：

- Node.js 20+
- Python 3.10+
- FFmpeg

不需要配置 GPT-SoVITS、CosyVoice 3 和 env_parse。  
只能打开已有项目或查看 UI 和基础交互，不能真正解析视频和生成配音。

---

### 情况 B：有 NVIDIA 显卡，8GB 左右，例如 RTX 5060 Laptop

这是当前开发机配置，推荐：

- 使用本地 pinned `MOSS-Transcribe-Diarize`（BF16 / SDPA）；最长 60 秒已经在 8GB RTX 5060 Laptop 上实测，详见 [MOSS 实测](docs/moss_setup.md)；
- `MOSS_MAX_NEW_TOKENS=65536` 是生成 token 上限，不是模型的 90 分钟官方输入能力；应用的 90 分钟输入预算独立配置；
- GPT-SoVITS 使用本仓库的 `backend/tts_v2.yaml`；
- 模型缓存放到非系统盘。

不要在此 MOSS 环境中混装或自动回退到 WhisperX / pyannote。

---

### 情况 C：显存 12GB 或更多

MOSS 默认允许最长 90 分钟。更长音频的显存、速度和结果完整性仍需要实测，8GB 显卡可能无法处理完整 90 分钟。

---

### 情况 D：没有 NVIDIA GPU / 只有 CPU

可以启动后端和前端，但不建议跑完整流水线：

- Banquet 会使用 CPU，长视频分轨耗时较长；
- MOSS 支持明确的 `cpu` 路径，但吞吐量与可处理长度需自行验证；
- GPT-SoVITS 实时合成基本不可用。

建议这种电脑只做前端开发、文档、接口 mock 或非实时测试。

---

### 情况 E：C 盘空间不足

把所有大文件都改到 D/E/F 盘：

```env
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache
```

`env_parse` 也可以放到非 C 盘。  
**不要把模型权重、缓存、env_parse 放进 C 盘。**

---

## 5. 详细环境安装步骤

下面以 Windows PowerShell 为主。  
如果你用 macOS / Linux，命令不同，但原则一样：**四个环境分开，路径写进 `.env`。**

---

### 5.1 安装 FFmpeg

#### Windows：一键脚本自动下载并配置（推荐）

在项目根目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1
```

脚本会自动完成：

1. 下载 FFmpeg Essentials Windows 构建；
2. 解压到：

```text
%LOCALAPPDATA%\StageTTS\ffmpeg
```

3. 把 `ffmpeg.exe` / `ffprobe.exe` 所在目录加入**当前用户 PATH**；
4. 立即在当前终端验证 `ffmpeg -version` 与 `ffprobe -version`。

如果你想装更大的 full build，或指定安装目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1 -Full -InstallDir F:\Tools\ffmpeg
```

如果系统里已经有 FFmpeg，但你想强制重装：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg_windows.ps1 -Force
```

验证：

```powershell
ffmpeg -version
ffprobe -version
```

如果能输出版本号，说明成功。

#### Windows：手动安装

也可以手动安装：

1. 下载 FFmpeg Windows build；
2. 解压到例如：

```text
C:\Tools\ffmpeg\bin
```

3. 把该目录加入系统 PATH；
4. 新开 PowerShell 验证：

```powershell
ffmpeg -version
ffprobe -version
```

如果能输出版本号，说明成功。

#### 常见问题

- 如果 PowerShell 提示找不到 `ffmpeg`，通常是 PATH 没生效，重启终端或重新登录系统。
- 后端和 worker 都会调用系统 PATH 里的 `ffmpeg` / `ffprobe`，不是某个 Python 包里的。

---

### 5.2 安装 Node.js 与 pnpm

推荐 Node.js 20 LTS（前端使用 `pnpm` 管理依赖与 `pnpm-lock.yaml`）。

验证：

```powershell
node -v
pnpm -v
```

若尚未安装 `pnpm`，可先启用 Corepack 或全局安装：

```powershell
corepack enable
# 或 npm install -g pnpm
```

然后进入前端目录安装依赖：

```powershell
cd frontend
pnpm install
```

---

### 5.3 配置主后端 env_backend

主后端只跑 FastAPI、文件读写、任务调度，**不直接跑 GPU 模型**。

在项目根目录创建独立 Python 环境：

```powershell
python -m venv .venv_backend
.\.venv_backend\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

复制环境变量模板：

```powershell
copy backend\.env.example backend\.env
```

然后编辑 `backend\.env`。

#### backend/.env 重点字段

```env
# 项目工作区，留空默认使用仓库内 workspace/
WORKSPACE_DIR=

# 静态资源目录，留空默认使用仓库内 assets/
ASSETS_DIR=

# GPT-SoVITS 安装根目录
GPT_SOVITS_ROOT=C:/Project/GPT-SoVITS-v5-20261006nvidia50x0

# 轻量 v2 CUDA 配置
GPT_SOVITS_CONFIG=backend/tts_v2.yaml

# 自动管理 TTS 进程
TTS_AUTO_MANAGE=1

ENV_PARSE_PYTHON=F:/Download/StageTTS/env_moss/python.exe
PARSE_BACKEND=moss
MOSS_MODEL_PATH=F:/Download/StageTTS/diarization_models/moss-transcribe-diarize
MOSS_REVISION=10304d926740cf5a5cee05a268086e3265197910
MOSS_TORCH_VERSION=2.8.0+cu128
MOSS_TRANSFORMERS_VERSION=5.6.0
MOSS_CUSTOM_CODE_APPROVED=0 # 仅在本地固定官方代码已审阅并核验 hash 后设置为 1
MOSS_MAX_AUDIO_SECONDS=5400
MOSS_MAX_NEW_TOKENS=65536
BANQUET_ROOT=F:/Download/StageTTS/banquet/bandit-v2
BANQUET_CHECKPOINT=F:/Download/StageTTS/banquet/banquet-split.ckpt
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache

PARSE_TIMEOUT_SEC=1800
LLM_TIMEOUT_SEC=60

# LLM 配置（示例值；推荐在 UI 顶部「设置」页配置）
LLM_API_KEY=local-gateway-no-key
LLM_MODEL=workbuddy-ai/hy4flash-free
```

The UI settings page persists model, endpoint and key in the ignored local backend settings file; keep secrets out of tracked documentation. Do not put access tokens in this example.

#### 在 UI 中选择配音引擎与 GPT-SoVITS 微调方案

顶部“设置”支持切换默认配音引擎（**GPT-SoVITS** 或 **CosyVoice 3（零样本）**，保存至忽略跟踪的 `backend/tts_settings.local.json`，切换时会自动停止已运行的旧引擎释放显存）；在 `GPT-SoVITS` 下展开折叠选项即可选择 `v2` / `v2Pro` / `v2ProPlus` 底模和 GPT/SoVITS 训练轮数（保存至 `backend/training_settings.local.json`），并在左侧「角色面板」直接发起单角色微调训练与音色预设管理。

#### 在 UI 中切换翻译模型

点击顶部“设置”，在“翻译与 LLM”中修改模型名称（服务商提供的模型 ID）、OpenAI 兼容接口根地址和 API Key，然后点击“保存设置”。只换模型时 API Key 留空即可沿用；清除 Key 需要显式勾选。保存不调用模型，也不验证服务商是否支持该模型。

设置保存到后端私有的 `backend/llm_settings.local.json`（已加入 Git 忽略，含明文 Key，不在媒体静态目录中），优先于环境默认值，重启后继续生效；前端不会收到旧 Key，也不会写入浏览器存储。新发起的解析翻译、重译、时长改写和 AI 助手请求共用这套配置；已运行的请求继续使用原配置。

#### 注意

- **不要提交 `backend/.env`**，它已在 `.gitignore` 中。
- 如果没有本地 LLM 网关，可以填任何 OpenAI-compatible 服务，或者暂时不配置；但不配置时 Stage4 不会翻译，只会保留原文。
- **翻译模型设置**：顶部“设置”页或 UI；旧 `.env` 示例仅用于首次默认配置，不得提交密钥。

---

### 5.4 配置配音引擎 env_tts（GPT-SoVITS 与可选 CosyVoice 3）

#### 选项 A：GPT-SoVITS（默认，支持角色微调）

使用适配 RTX 50 系列 / CUDA 12.8 的 GPT-SoVITS 整合包。  
开发机路径示例：

```text
C:\Project\GPT-SoVITS-v5-20261006nvidia50x0
```

确认它能找到 `runtime\python.exe` 和 `api_v2.py`，并在 `backend/.env` 中配置：

```env
GPT_SOVITS_ROOT=C:/Project/GPT-SoVITS-v5-20261006nvidia50x0
GPT_SOVITS_CONFIG=backend/tts_v2.yaml
TTS_AUTO_MANAGE=1
```

#### 选项 B：CosyVoice 3（零样本，端口 `9881`）

适合免训练直接零样本克隆。在独立 Python 3.10 + CUDA 12.8 环境部署 `Fun-CosyVoice3-0.5B-2512`，并在 `backend/.env` 中配置：

```env
COSYVOICE3_ROOT=F:/Download/StageTTS/CosyVoice
COSYVOICE3_PYTHON=F:/Download/StageTTS/env_cosyvoice3/Scripts/python.exe
COSYVOICE3_MODEL_DIR=F:/Download/StageTTS/cosyvoice_models/Fun-CosyVoice3-0.5B-2512
COSYVOICE3_API_URL=http://127.0.0.1:9881
```

由 `workers/cosyvoice3_server.py` 托管启动，采用 `float16` 初始化与流式显存加载（RTX 5060 8GB 实测占用约 2.0 GB 显存），并在跨语种（如日文参考音 → 中文台词）时自动切换 `inference_cross_lingual` 消除外语口音。

开启 `TTS_AUTO_MANAGE=1` 后，后端会根据当前所选引擎按需自动拉起对应 TTS 服务。

#### 如果要手动启动 GPT-SoVITS

```powershell
cd /d C:\Project\GPT-SoVITS-v5-20261006nvidia50x0
.\runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c c:\Project\StageTTS\backend\tts_v2.yaml
```

#### 手动启动时的注意事项

当前 `TTS_AUTO_MANAGE` 只能管理“后端自己拉起”的 GPT-SoVITS。  
如果你手动启动了 9880，后端解析前不会自动杀掉这个外部进程。

所以：

- 如果要跑解析，建议先手动关闭 9880；
- 解析完成后，再启动 9880 配音；
- 或者不要手动启动，交给后端自动管理。

---

### 5.5 配置解析环境 `env_parse`

Banquet 四轨分离和 MOSS 主解析共享一个独立 CUDA 子进程环境；不安装 WhisperX 或 pyannote 作为默认路径，也不把 GPU/ML 包装进 `env_backend`。

#### Banquet 四轨分离

1. 在本机检出 [bandit-v2](https://github.com/kwatcharasupat/bandit-v2) 已审查提交 `d5563d9031`，并记录实际 commit/hash。
2. 单独取得 Netflix 官方 `banquet-split.ckpt`，核对 MD5 `4ac581d6792740f5b7be8e0a3f73c8ea`。
3. 配置 `BANQUET_ROOT` 和 `BANQUET_CHECKPOINT`。Bandit 依赖须与 MOSS 的 torch/torchaudio ABI 相容；**不要用旧 Banquet requirements 将 torch 2.8 降级**。

#### MOSS 依赖、模型与配置

见 [MOSS 设置与验证文档](docs/moss_setup.md)。关键固定信息：Python 3.12、torch/torchaudio 2.8.0+cu128、transformers 5.6.0；MOSS revision 与权重 SHA-256 固定在该文档。先安装 CUDA 12.8 torch 栈，再安装 `workers/requirements-moss.txt`；隔离环境设为 `ENV_PARSE_PYTHON`，backend `.env` 配置 `PARSE_BACKEND=moss`、本地 `MOSS_MODEL_PATH`、固定 revision 与代码审批标志。

MOSS 及应用默认输入上限为 90 分钟。最长 60 秒样例仅代表已完成的资源实测范围。输入/输出共享有限上下文；独立窗口的匿名说话人不能直接认作同一人。

#### 缓存目录

```env
HF_HOME=F:/Download/StageTTS/hf_cache
HUGGINGFACE_HUB_CACHE=F:/Download/StageTTS/hf_cache/hub
TORCH_HOME=F:/Download/StageTTS/torch_cache
PIP_CACHE_DIR=F:/Download/StageTTS/pip_cache
```

推理只读取经批准的本地 snapshot，不会静默下载模型。

---

---

#### MOSS/Banquet 依赖安装注意

Banquet 来自第三方 Bandit-v2 实现，必须在**同一个 `env_parse` Python 环境**运行，以满足子进程 torch ABI。其旧官方锁文件针对 torch 2.0/CUDA 11.8；不能照抄整份 requirements 覆盖 MOSS 的 torch 2.8/CUDA 12.8 栈。当前开发环境已在上述固定 torch 环境验证短 Banquet CUDA stem smoke；新机器需按绑定的 Bandit commit 检查缺失 Python 包与版本兼容，禁止降级 / 替换 torch、torchaudio 和 transformers。Banquet 权重与 Bandit 源码由用户自行取得并校验，仓库不分发。

---

---

## 6. 启动项目

### 6.1 标准启动：2 个终端

#### 终端 1：启动后端

在项目根目录：

```powershell
.\.venv_backend\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

或者直接：

```powershell
F:\Python\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

验证：

浏览器打开：

```text
http://127.0.0.1:8000/api/health
```

应该看到：

```json
{
  "status": "ok"
}
```

#### 终端 2：启动前端

```powershell
cd frontend
pnpm dev
```

浏览器打开：

```text
http://127.0.0.1:5173/
```

---

## 7. 第一次使用流程

1. 打开 `http://127.0.0.1:5173/`
2. 如果右上角显示“后端已连接”，说明前后端连通。
3. 导入音视频或打开最近项目：
   - 点“导入媒体”或将 MP4 / WAV 拖入启动框创建项目；
   - 或从右侧“最近项目”列表直接打开。
4. 切到顶部“节点流”：
   - 画布中各节点卡片直接内嵌了核心参数（如「翻译与情绪」内的目标/原声语言、**口语润色**、**联网校对**开关；「混音导出」内的伴奏音量、**字幕烧录（无 / 原文（识别） / 译文 / 双语）**、闪避与水印）；
   - 在画布选中「媒体输入 / 音轨分离」开始 Banquet + MOSS 完整解析；已有对白时可选中「翻译与情绪」节点仅执行翻译；选中「剧本枢纽 / 语音合成 / 混音导出」可从对应断点继续；
   - 右侧“运行日志”显示实际执行步骤，跳过的节点保持待机；支持“暂停 / 继续 / 结束流程”。
5. 切回“剪辑”微调与配音：
   - 时间轴支持**拖动块体整体平移片段**、拖拽左右边缘微调入/出点，可在时间轴工具栏点「全部确认」或左侧角色面板点「一键确认本角色」批量消除待校对状态；
   - 单句可点“重配”，全部未就绪片段可点时间轴右上“合成待处理”。
6. 满意后在「导出」弹窗或「混音导出」节点导出成片（支持烧录识别原文/译文/双语 ASS 硬字幕与同名外挂 `.ass`）。

---

## 8. 常见问题排查

### Q1：前端显示“离线预览模式”

说明后端没启动或端口不对。检查：

```text
http://127.0.0.1:8000/api/health
```

如果后端改了端口，需要同步修改 `frontend/vite.config.ts` 里的代理目标。

---

### Q2：`ffmpeg` 找不到

验证：

```powershell
ffmpeg -version
```

如果失败，说明 FFmpeg 没加 PATH。

---

### Q3：解析时 GPT-SoVITS 占了显存

如果你手动启动了 9880，解析前请先关掉它。  
当前后端只能关闭自己自动拉起的 TTS 进程，不能强制关闭用户手动启动的进程。

---

### Q4：MOSS 占用显存或解析失败

MOSS 使用 BF16 / SDPA，短样例已在 RTX 5060 8GB 验证；系统会在解析前停止由后端托管的 GPT-SoVITS。关闭其他 GPU 程序；不要在模型初始化或解析任务仍运行时重启 backend。 MOSS 配置 / 权重失败时检查 `MOSS_MODEL_PATH`、固定 revision 和本地 hash manifest；应用时长超过配置的输入上限时会 fail-fast。

---

### Q5：MOSS 输入时长限制

默认 MOSS 输入上限为 5400 秒（90 分钟）。如仍提示 60 秒超限，请确认本地配置为 `MOSS_MAX_AUDIO_SECONDS=5400` 并重启后端。输出预算为 `MOSS_MAX_NEW_TOKENS=65536`，实际还受模型剩余上下文和硬件资源限制。

---

### Q6：9880 根路径返回 404

GPT-SoVITS 的 `/` 返回 404 不代表不可用。  
只要 `/tts` 能正常返回音频即可。

---

### Q7：改了台词但音频没变

检查是否重新合成。  
`target_text`、`speaker`、`emotion`、`target_lang`、`keep_original` 修改后，片段应变为 `pending`，需要重新合成。

前端已有音频防缓存机制，但前提是重新合成后 `tts_wav` 被回写。

---

## 9. 仓库目录结构

```text
StageTTS/
├── ARCHITECTURE.md                     # 架构、铁律、接口契约
├── FRONTEND_DESIGN.md                  # 前端设计规范
├── README.md                           # 本文件
├── backend/
│   ├── main.py                         # FastAPI 入口
│   ├── config.py                       # 路径、环境、缓存、调度配置
│   ├── schemas.py                      # script.json 数据契约
│   ├── tts_v2.yaml                     # 8GB 显卡轻量 GPT-SoVITS 配置
│   ├── .env.example                    # 环境变量模板
│   ├── agent/                          # 导演/润色 Prompt 与编排工具
│   └── services/                       # TTS 路由/设置、三步翻译、联网检索、微调、混音、字幕、GPU 调度
├── workers/
│   ├── parse_worker.py                 # Banquet + MOSS 解析流水线
│   ├── banquet_worker.py               # 四轨分离 worker
│   └── cosyvoice3_server.py            # CosyVoice 3 低显存零样本推理服务
├── frontend/
│   └── src/
│       ├── components/                 # UI 组件
│       ├── store/                      # 前端状态管理
│       ├── services/api.ts             # 后端接口封装
│       └── types/                      # TypeScript 类型
├── experiments/                        # 评测脚本骨架
├── assets/                             # 静态资源
└── workspace/                          # 运行时项目目录
```

---

## 10. 评测计划

评测脚本在 [experiments/](experiments)，目前是骨架：

- `eval_diarization.py`：Banquet 分离前后 WER/CER、DER 对比
- `eval_batch_speed.py`：按 speaker 分组合成 vs 逐句合成耗时/显存对比
- `eval_speaker_sim.py`：声纹相似度与 MOS 评测

后续建议补充：

- `eval_duration_reflection.py`：时长反思前后超时率
- `eval_e2e.py`：端到端成功率、阶段耗时、显存峰值

评测应该回答：

1. 解析更准了吗？
2. 剧本更自然了吗？
3. 配音更像了吗？
4. 超长台词真的变短了吗？
5. 8GB 显卡下更稳了吗？

---

## 11. AI 伦理与合规声明

- 本项目仅限使用团队成员授权音色或开源无版权许可音色。
- 严禁未经授权克隆商业声优、影视角色或公众人物声音。
- 所有导出视频与虚拟演播室界面均需保留“AI 合成语音”显式标识。
- 禁止用于欺诈、冒充、误导传播等不当用途。


### 3.5 M1 / M2 / M3 可选源声证据与分析

新解析可在 `script.evidence` 保存独立源证据：原始ASR整句与字词、可重叠的diarization活动区间、源turn，以及翻译/表演单位的来源链接。exclusive轨是派生方便轨，不替代真实重叠活动。旧JSON缺少evidence时仍可打开，不能从旧TTS倒造源声证据。

M2边界候选仅来自原文字符范围和已有alignment。分词器可注入，缺少JA/ZH词典时保守回退待审；可选LLM只能选择已有候选ID，不能发明时间戳或词中切点。默认主解析为固定 revision 的 MOSS joint ASR + 窗口内匿名角色 + 句段时间，使用独立 MOSS / Banquet 环境，旧 WhisperX / pyannote worker 只保留代码以供单独审查的 legacy 实验，不作为依赖清单、默认入口或自动 fallback。官方原精度BF16+SDPA已测JA/EN10、30、完整60秒单scope，当前默认输入上限5400秒/65536输出token；超出配置输入预算明确拒绝，不能外推90分钟或自动切窗认同S01。MOSS不提供word时间，不按字符均分；`alignment_granularity`与独立句段审核保留这种能力差别。整段原文/边界与角色确认后可用完整3–10秒参考，超过10秒无词级对齐不允许裁剪。官方固定snapshot与本地custom code已hash验证，community-1暂停。

源声韵律/情绪分析为显式操作，不跟随每次保存自动运行。精修面板显示证据链接与分析状态；`not_run`/`not_configured`/`blocked`不是neutral。CPU能量/活动区间等测量与模型情绪分数分开保存，原始分数不冒充校准概率，不由标签猜F0。GPT-SoVITS当前只应用参考路由；CV3/Index适配状态不代表已接通。

情绪分析使用独立解释器和经许可确认的本地权重：`ENV_ANALYSIS_PYTHON`、`EMOTION_MODEL_PATH`、`EMOTION_MODEL_ID`、`EMOTION_REVISION`、`EMOTION_FUNASR_VERSION`，以及默认关闭的`EMOTION_USER_AUTHORIZED`。不要更改现有 CUDA / PyTorch 运行环境；情绪分析与 MOSS 主解析分别保留独立解释器，按各自锁文件部署。若仅运行 MOSS，则参阅 [MOSS 安装与验证文档](docs/moss_setup.md)。
