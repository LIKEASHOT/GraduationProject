# 全双工语音通话后端接口需求规范

本规范定义了为配合 Android App（HBuilder uni-app 打包环境）以及 Qwen1.5B 等流式大模型，所需的**全双工 WebSocket 通信协议**。

## 1. 核心架构与协议概览

- **通信协议**：WebSocket
- **连接路径**：`ws://HOST:PORT/ws/full-duplex`
- **数据格式**：JSON 字符串。音频数据使用 Base64 编码内嵌。
- **音频约定**：16000Hz 采样率、单声道、16bit PCM 裸流。
- **VAD 架构**：**前端本地运行 Silero VAD v5**（ONNX），负责检测用户语音的起止。后端不再需要做 VAD 断句，只需接收完整语音段。前端发送 `speech_end` 事件通知后端一句话说完，后端即可开始 ASR → LLM → TTS 处理。

## 2. 上行链路 (Client -> Server)

前端 VAD 检测到用户开始说话后，在 `onSpeechEnd` 时将完整语音段一次性发送给后端。

### 2.1 推流音频块 (Audio Stream)
客户端将采集到的录音片段用 Base64 编码发给后端。

```json
{
  "event": "audio_stream",
  "payload": {
    "audio_chunk": "UklGRmR5AABXQVZFZm10IBAAAAABAAEA...", 
    "sample_rate": 16000,
    "format": "pcm" 
  }
}
```

### 2.2 语音结束通知 (Speech End)
前端 VAD 检测到用户停止说话后发送此事件。**后端收到此事件后，应立即开始 ASR → LLM → TTS 处理流水线。**

```json
{ "event": "speech_end" }
```

### 2.3 打断通知 (Client Interrupt)
用户在 AI 说话时再次开口，前端 VAD 自动检测到人声并发送打断：
```json
{ "event": "interrupt", "payload": { "reason": "user_button" } }
```

### 2.4 其他控制信号
```json
// 麦克风静音
{ "event": "mute_mic", "payload": { "status": true } }

// 挂断通话
{ "event": "end_call", "payload": {} }
```

---

## 3. 下行链路 (Server -> Client)

服务端收到 `speech_end` 后，处理管线为：**收到的完整 PCM 音频 → ASR 转文字 → Qwen1.5B 流式生成 → TTS 语音合成**。

### 3.1 识别文本回调 (ASR Text)

```json
{
  "event": "user_text",
  "payload": {
    "text": "我想点一杯咖啡。", 
    "is_final": true
  }
}
```

### 3.2 大模型文字推送 (LLM TextStream)

```json
{
  "event": "ai_text",
  "payload": {
    "text": "好的，没问题！", 
    "is_final": false,
    "message_id": "msg_001"
  }
}
```

### 3.3 语音推流 (TTS Audio Stream)

```json
{
  "event": "tts_audio_chunk",
  "payload": {
    "audio": "qwe28f...",
    "format": "pcm", 
    "sample_rate": 16000,
    "chunk_id": "1",
    "is_last_chunk": false 
  }
}
```

---

## 4. 打断逻辑 (Interruption)

**前端 VAD 同时承担打断检测**。当 AI 正在通过 TTS 播放回复时，如果前端 VAD 检测到用户再次开口说话：

1. **前端自动行为**：
   - 立即清空 TTS 播放队列，停止当前音频播放。
   - 向后端发送 `{ "event": "interrupt" }` 通知。
   - 新语音段照常录制，VAD 结束后发送新的 `audio_stream` + `speech_end`。

2. **后端收到 `interrupt` 后应**：
   - 立即中止当前正在进行的 LLM 生成和 TTS 合成。
   - 丢弃未发送的 TTS 音频队列。
   - 等待接收前端新的 `audio_stream` + `speech_end` 开始新一轮处理。

3. **后端也可主动下发 interrupt**：
```json
{ "event": "interrupt", "payload": { "reason": "user_speaking", "stop_audio_playback": true } }
```

---

## 5. 后端技术选型建议

**WebSocket + 级联 (ASR → Qwen1.5B → TTS)** 模式：
- **Web 框架**：FastAPI + asyncio
- **VAD**：前端已用 Silero VAD v5 完成断句，后端无需再做 VAD。后端只需监听 `speech_end` 事件即可触发 ASR。
- **ASR**：使用 FunASR/Whisper 等，接收完整 PCM 段做识别。
- **LLM**：Qwen1.5B/7B 流式输出。
- **TTS**：流式合成后分块下发 `tts_audio_chunk`。
