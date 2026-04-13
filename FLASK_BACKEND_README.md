# 全双工语音通话后端接口需求规范

本规范定义了为配合 Android App（HBuilder 打包环境）以及 Personaplex-7B 等流式大模型，所需的**全双工 WebSocket 通信协议**。传统的 HTTP 无法满足同时录音推流和播音放流的特性，必须建立全双工双向长连接。

## 1. 核心架构与协议概览

- **通信协议**：WebSocket (WSS 推荐用于生产环境)
- **连接路径 (示例)**：`wss://api.yourserver.com/ws/full-duplex?token=USER_TOKEN`
- **数据格式**：全部包裹为 `JSON` 字符串格式发送。内部包含针对大体积数据的 `Base64` 编码。
- **编码约定**：
  - 音频：建议统一采用 `16000Hz` 采样率的单声道 `.pcm`裸流 或 `.mp3` 块。
  - 文本：UTF-8 编码。

## 2. 上行链路 (Client -> Server)

客户端（App）在连接成功后，一旦 VAD（声音端点检测）检测到用户开始说话，即循环密集地（约每 50-100ms）向服务端发送切片。

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

### 2.2 客户端控制信号
用户主动关闭麦克风或结束对话。

```json
// 麦克风静音
{ "event": "mute_mic", "payload": { "status": true } }

// 挂断通话
{ "event": "end_call", "payload": {} }
```

---

## 3. 下行链路 (Server -> Client)

在“级联模拟全双工”的架构下，服务端的处理管线应为： **循环接收 PCM 流 -> VAD 感知到停顿（断句） -> 送给 ASR 转文字 -> 送入 Qwen1.5B (流式生成) -> TTS (语音合成)**。

### 3.1 识别文本回调 (ASR Text)
一旦 VAD 检测到用户说完了当前句子并进行 ASR 处理后，需把人类说出来的最终文本回传给前端显示。

```json
{
  "event": "user_text",
  "payload": {
    "text": "我想点一杯咖啡。", 
    "is_final": true // 由于是 VAD 断句的级联，发过来的通常直接是 true 完整句
  }
}
```

### 3.2 大模型文字推送 (LLM TextStream)
Qwen1.5B 开始流式生成时，逐个推送部分生成的文字，用于前端极速上屏。

```json
{
  "event": "ai_text",
  "payload": {
    "text": "好的，没问题！", 
    "is_final": false,     // Qwen是否生成完毕
    "message_id": "msg_001" // 确保属于同一组对话
  }
}
```

### 3.3 语音推流 (TTS Audio Stream)
将大模型生成的半段文字（或全部文字）交给 TTS 并下发 PCM 二进制。**前端拿到后使用 AudioContext 拼合播放。**

```json
{
  "event": "tts_audio_chunk",
  "payload": {
    "audio": "qwe28f...", // TTS生成的 PCM/Wav Base64
    "format": "pcm", 
    "sample_rate": 16000,
    "chunk_id": "1",
    "is_last_chunk": false 
  }
}
```

---

## 4. 关键体验逻辑：级联下的“假打断” (Interruption handling)

在级联架构下，VAD不仅用来判断用户是否说完了当前的完整一句话，**还负责打断AI的声音**。
当服务端**正在通过 TTS 下发音频流给客户端播放 AI 说话**时，由于客户端的麦克风是一直在采音上行的，如果此时后端的 VAD 引擎突然识别到**用户传来的上行音频中出现了清晰的讲话人声（用户强行插嘴）**，请立即打断！

### 处理流程：
1. **服务端动作**：
   - 立即掐断/丢弃队列中尚未下放完毕的 Qwen1.5B 文本输出和 TTS 合成流程。
   - 向前端下发 `interrupt` 信号，让前端立刻闭嘴（停止播放）。
   - 将刚才收到的新声音作为下一轮 ASR 级联的新起点。
   
2. **客户端侧接收结构**：
```json
{
  "event": "interrupt",
  "payload": {
    "reason": "user_speaking",
    "stop_audio_playback": true
  }
}
```

3. **客户端行为**：
   - 收到该协议后，客户端直接停止正在播放的声音（切除所有音频队列）。
   - UI 会在新的一行立刻开启最新一轮用户对话表现。

---

## 5. 后端技术选型建议

针对目前敲定的 **WebSocket 流 + 级联 (ASR->Qwen1.5B->TTS)** 模式：
- **核心组件 VAD**：必须极为健壮，推荐极速级的 `Silero VAD`。它是整个流式大管线的**总调度员**，它必须敏感判断用户的断句（何时该喂给 ASR），且敏锐判断插嘴（何时该掐断当前的 TTS）。
- **Web 服务框架**：推荐 `FastAPI` 配合原生 `asyncio`，通过异步队列分别维护：
  1. 上行音频流收集任务
  2. VAD 分发中心任务 
  3. Qwen 推理+TTS下发管道。
