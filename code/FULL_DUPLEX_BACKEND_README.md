# EchoSage 实时假全双工后端

这套新增代码不会替换现有 Flask 接口。
现在推荐通过单端口统一入口运行。

- 原有文字聊天：继续走 `/api/chat`
- 原有单轮语音识别：继续走 `/api/speech-to-text`
- 原有单轮 TTS：继续走 `/api/text-to-speech`
- 新增假全双工：走 `/ws/full-duplex`

## 设计原则

- 保留成熟的文字聊天和原有语音链路
- 新增一条独立的实时会话通道，避免互相影响
- 路线固定为 `前端VAD -> ASR -> LLM -> TTS`
- 前端负责本地 VAD、断句和插话检测
- 后端只缓存整段音频，收到 `speech_end` 后开始处理
- 前端插话时主动发送 `interrupt`
- 会话历史只保留已经开始下发到前端的 TTS 文本片段

## 主要文件

- `asgi_app.py`
  - FastAPI 入口
- `full_duplex_session.py`
  - WebSocket 会话状态机和打断调度
- `full_duplex_backends.py`
  - SenseVoice / Qwen / CosyVoice / TTS 适配层
- `full_duplex_vad.py`
  - 保留旧版后端 VAD 实验封装，目前实时主流程不再依赖它

## 默认模型策略

- ASR：实时链路强制使用 SenseVoice，不再回退到 Whisper
- LLM：优先本地 `Qwen2.5-1.5B-Instruct`
- TTS：优先 CosyVoice，旧 Flask 链路和实时链路现在都共用这个优先策略
- VAD：实时主流程已迁移到前端，后端不再负责断句

## 启动方式

统一启动单端口后端：

```powershell
cd code
uvicorn asgi_app:app --host 0.0.0.0 --port 8000
```

## 当前实现边界

- 这是“保留旧链路 + 新增实时链路”的升级版
- 目前已经具备前端驱动断句、前端驱动打断、句段 TTS、会话状态管理
- 如果本地缺少 SenseVoice 依赖，实时服务会直接启动失败并给出明确错误
- 若后续前端补充播放 ACK，可以把“只保留已播文本”从“已下发文本”进一步收紧到“已实际播放文本”
