# Full Duplex Backend Notes

后端统一使用单端口 ASGI 服务，同时承载原有 Flask HTTP 接口和全双工 WebSocket。

## 启动方式

```powershell
cd "E:\Desktop\Graduation Project\code"
uvicorn asgi_app:app --host 0.0.0.0 --port 8000
```

## 保留的原有接口

- `/api/chat`：文字聊天继续可用，`need_audio=true` 时仍会生成语音文件。
- `/api/speech-to-text`：原语音识别接口继续可用，ASR 后端为 SenseVoice。
- `/api/text-to-speech`：原独立 TTS 接口继续可用，统一走 `TTSEngine`。
- `/upload`：文件上传后返回文件信息，同时尝试做一次 ASR 并返回 `text`。

## 全双工接口

- `/ws/full-duplex`：全双工 WebSocket 入口。
- 前端负责 VAD，后端收到 `speech_end` 后提交一句话进入级联链路。
- 级联链路为 `SenseVoice ASR -> Qwen LLM -> TTSEngine`。
- 用户插话时前端发送打断事件，后端会停止当前生成任务，并只把已经实际播出的 assistant 文本计入历史。

## 模型配置

- ASR：SenseVoice，默认路径 `models/SenseVoiceSmall`，也可用 `SENSEVOICE_MODEL_PATH` 覆盖。
- LLM：Qwen2.5-1.5B-Instruct，默认路径 `models/Qwen2.5-1.5B-Instruct`，也可用 `REALTIME_QWEN_MODEL_PATH` 覆盖。
- TTS：优先在后端进程内常驻加载 MOSS-TTS-Nano，失败后才尝试 Edge TTS / gTTS 网络兜底。

## MOSS-TTS-Nano 常驻接入

后端直接把 MOSS-TTS-Nano 加载到当前进程，并在启动后常驻内存/显存。这样每句 TTS 不再重新启动进程，也不会重复加载模型，优先保证全双工速度。

推荐本地模型配置：

```powershell
$env:MOSS_TTS_MODEL_PATH="E:\Desktop\Graduation Project\models\MOSS-TTS-Nano"
```

如果不使用本地模型目录，也可以指定模型名并允许在线加载：

```powershell
$env:MOSS_TTS_MODEL_NAME="OpenMOSS-Team/MOSS-TTS-Nano"
$env:MOSS_TTS_LOCAL_FILES_ONLY="0"
```

常用可选项：

- `MOSS_TTS_DEVICE`：默认自动选择 `cuda` 或 `cpu`。
- `MOSS_TTS_MAX_NEW_FRAMES`：默认 `375`。
- `MOSS_TTS_AUDIO_TEMPERATURE`：默认 `1.0`。
- `MOSS_TTS_REFERENCE_AUDIO`：可选参考音频路径。
- `MOSS_TTS_EAGER_LOAD`：默认 `1`，启动时加载模型；设为 `0` 时第一次合成再加载。

仅作为应急兜底的外部进程模式，默认不开启：

```powershell
$env:ENABLE_MOSS_TTS_PROCESS_FALLBACK="1"
$env:MOSS_TTS_REPO_PATH="E:\Desktop\Graduation Project\MOSS-TTS-Nano"
$env:MOSS_TTS_PYTHON="C:\Users\test\miniconda3\envs\moss_tts\python.exe"
```

## 预期日志

启动时：

```text
TTS engine config - MOSS-TTS-Nano first: True, Edge fallback: True, gTTS fallback: True
Loading MOSS-TTS-Nano direct model: ...
MOSS-TTS-Nano direct model loaded
[full_duplex] Realtime TTS engine: MOSS-TTS-Nano local
```

合成成功时：

```text
[OK] MOSS-TTS-Nano direct speech file generated: ...
```
