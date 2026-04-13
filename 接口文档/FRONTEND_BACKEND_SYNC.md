# 前端对接说明

这份文档专门给前端同学看，说明这次后端升级已经完成了什么，以及前端接下来应该如何对接。

## 这次后端改了什么

后端这次不是推翻原来的 Flask 方案重写，而是在保留现有成熟能力的前提下，新增了一套“假全双工”实时语音通道，并且已经整理成单端口入口。

也就是说，现在项目里有两条后端链路并存，但前端只需要连一个端口：

- 旧链路：Flask HTTP 接口
- 新链路：FastAPI WebSocket 实时接口

当前推荐统一入口：

- HTTP: `http://127.0.0.1:8000`
- WebSocket: `ws://127.0.0.1:8000/ws/full-duplex`

## 保留不变的能力

下面这些现有能力都保留，没有被删，也没有被替换：

- 文字聊天接口仍然保留
- 文字聊天中原有的 TTS 能力仍然保留
- 原有单轮语音识别接口仍然保留
- 原有 Flask 后端仍然可以继续启动和使用

对应旧接口仍然是：

- `/api/chat`
- `/api/speech-to-text`
- `/api/text-to-speech`

所以前端现有“文字聊天模式”不需要因为这次升级而废弃。

## 新增的能力

后端新增了一套实时假全双工语音会话能力，核心目标是：

- 仍然采用级联路线：`VAD -> ASR -> LLM -> TTS`
- AI 说话时，前端仍然持续上传麦克风音频
- 后端检测到用户插话时，立即打断当前 AI 播放
- 然后把用户新的语音作为下一轮输入继续处理

注意，这不是“真全双工”，而是“模拟全双工体验”的假全双工。

## 后端新增了哪些模块

后端新增了以下文件：

- [code/asgi_app.py](/e:/Desktop/Graduation%20Project/code/asgi_app.py)
  - 实时服务入口
- [code/full_duplex_session.py](/e:/Desktop/Graduation%20Project/code/full_duplex_session.py)
  - 单个 WebSocket 会话的状态机、打断逻辑、消息分发
- [code/full_duplex_vad.py](/e:/Desktop/Graduation%20Project/code/full_duplex_vad.py)
  - VAD 检测、打断判断、基础防回声误触策略
- [code/full_duplex_backends.py](/e:/Desktop/Graduation%20Project/code/full_duplex_backends.py)
  - 实时模式下 ASR / LLM / TTS 的适配层
- [code/FULL_DUPLEX_BACKEND_README.md](/e:/Desktop/Graduation%20Project/code/FULL_DUPLEX_BACKEND_README.md)
  - 后端内部说明文档

此外，共享能力也做了补充：

- [code/conversation_engine.py](/e:/Desktop/Graduation%20Project/code/conversation_engine.py)
  - 新增了流式文本生成能力
- [code/config.py](/e:/Desktop/Graduation%20Project/code/config.py)
  - 新增了实时模式参数

## 新实时接口

新增实时语音接口为：

- WebSocket 路径：`/ws/full-duplex`

它和旧的 Flask HTTP 接口是并行存在的，不冲突。

## 后端实时模式的工作方式

后端现在的实时语音会话流程是：

1. 前端建立 WebSocket 连接
2. 前端持续发送 `audio_stream`
3. 后端用 VAD 判断用户是否开始说话、是否说完
4. 用户说话过程中，后端可以返回中间识别文本
5. 用户一句结束后，后端提交最终 ASR 文本
6. 后端把最终文本送入 LLM
7. LLM 流式生成回复文本
8. 后端按句子或标点切段做 TTS
9. 后端分块下发音频给前端播放
10. 如果 AI 说话时检测到用户插话，后端立即发 `interrupt`

## 现在后端会发哪些实时事件

### 1. `session_ready`

用于告诉前端：连接已经建立，可以开始发音频。

示例：

```json
{
  "event": "session_ready",
  "payload": {
    "session_id": "session_ab12cd34"
  }
}
```

### 2. `vad_state`

用于调试或驱动 UI 状态，表示当前更像是“有人声”还是“静音”。

示例：

```json
{
  "event": "vad_state",
  "payload": {
    "state": "speech",
    "speech_prob": 0.83,
    "echo_similarity": 0.12
  }
}
```

### 3. `user_text_partial`

表示用户当前说话过程中的中间识别文本。

示例：

```json
{
  "event": "user_text_partial",
  "payload": {
    "text": "i want to",
    "is_final": false
  }
}
```

### 4. `user_text`

表示一句话结束后的最终识别结果。

示例：

```json
{
  "event": "user_text",
  "payload": {
    "text": "I want to order a coffee.",
    "is_final": true
  }
}
```

### 5. `ai_text`

表示大模型流式输出的文本。

示例：

```json
{
  "event": "ai_text",
  "payload": {
    "turn_id": "turn_0001",
    "message_id": "msg_0001",
    "text": "Sure, ",
    "is_final": false
  }
}
```

当一轮文本输出结束时，后端会再发一次：

```json
{
  "event": "ai_text",
  "payload": {
    "turn_id": "turn_0001",
    "message_id": "msg_0001",
    "text": "",
    "is_final": true
  }
}
```

### 6. `tts_audio_chunk`

表示一段 TTS 音频块。

示例：

```json
{
  "event": "tts_audio_chunk",
  "payload": {
    "turn_id": "turn_0001",
    "message_id": "msg_0001",
    "audio": "base64_pcm_here",
    "format": "pcm",
    "sample_rate": 16000,
    "chunk_id": "turn_0001_1",
    "is_last_chunk": false,
    "text_span": "Sure, I can help with that."
  }
}
```

说明：

- `audio` 是 Base64 编码的 PCM 数据
- `format` 当前默认是 `pcm`
- `sample_rate` 当前默认是 `16000`
- `text_span` 表示这段音频对应的文本

### 7. `interrupt`

表示后端判断用户在 AI 说话时插话了，前端应立刻停止当前播放。

示例：

```json
{
  "event": "interrupt",
  "payload": {
    "turn_id": "turn_0001",
    "reason": "user_speaking",
    "stop_audio_playback": true
  }
}
```

前端收到这个事件后应立即：

- 停止当前正在播的音频节点
- 清空还没播的音频队列
- 进入新一轮用户发言态

### 8. `turn_end`

表示一轮 AI 回复结束。

示例：

```json
{
  "event": "turn_end",
  "payload": {
    "turn_id": "turn_0001",
    "finish_reason": "completed"
  }
}
```

如果被打断，`finish_reason` 会是：

- `interrupted`

### 9. `error`

表示会话中出现错误。

## 前端需要发送哪些事件

### 1. `session_start`

可选，用于通知会话开始。

### 2. `audio_stream`

这是最核心的上行事件，前端需要持续发送音频块。

示例：

```json
{
  "event": "audio_stream",
  "payload": {
    "audio_chunk": "base64_pcm_here",
    "sample_rate": 16000,
    "format": "pcm"
  }
}
```

### 3. `mute_mic`

用于通知后端当前是否静音。

### 4. `end_call`

用于结束本次实时会话。

## 后端在“打断”这件事上已经做了什么

后端已经实现了下面这些能力：

- AI 播放期间仍然持续接收用户音频
- 基于 VAD 检测插话
- 触发插话后停止当前 LLM / TTS 这轮后续生成
- 发送 `interrupt` 给前端
- 当前轮 assistant 历史只保留已经开始下发到前端的文本片段

也就是说，后端现在已经能承担“谁来决定打断”这部分逻辑。

## 后端做了哪些防误触处理

因为当前不能假设前端有 AEC，所以后端加了基础防误触策略：

- 使用 VAD 概率而不是单帧裸判断
- 保留最近下发的 TTS 音频参考
- 对上行音频和最近 TTS 做基础相似度比较
- 在 TTS 刚开始下发的极短时间内做 grace window

这套机制目前是“可工作的第一版”，目标是优先保证灵敏度和整体交互通路。

## 前端现在最需要完成的工作

前端接下来重点是把实时播放调度器补齐。

建议前端最少完成下面这些能力：

- 建立 WebSocket 连接到 `/ws/full-duplex`
- 持续发送 `audio_stream`
- 接收 `user_text_partial` 和 `user_text` 更新 UI
- 接收 `ai_text` 做文本渐进显示
- 接收 `tts_audio_chunk` 并写入播放队列
- 接收 `interrupt` 时立刻停止当前播放并清空队列
- 接收 `turn_end` 后更新当前轮状态

## 关于文字聊天模式

这次升级没有要求前端放弃文字聊天。

建议前端继续保持双模式：

- 文字聊天模式：继续调用旧 Flask HTTP 接口
- 实时假全双工模式：走新的 WebSocket 实时接口

这样最稳，也最符合当前项目现状。

## 当前边界

目前这次后端升级的定位是：

- 保留老接口
- 新增实时接口
- 完成实时语音交互的后端主流程骨架

当前还没有做的事情包括：

- 前后端真实联调
- 前端播放 ACK 回传
- 更精细的回声抑制
- 更复杂的流式 TTS 优化

所以你可以把现在的状态理解为：

- 后端已经把“前端可接的实时协议”和“假全双工后端框架”搭好了
- 前端现在可以开始放心写 WebSocket 收发、PCM 播放队列和 `interrupt` 响应逻辑

## 一句话总结

后端没有砍掉你们现有的文字聊天方案，而是在其上新增了一套独立的实时假全双工语音通道；前端现在只需要把新的 WebSocket 音频收发和打断播放逻辑接起来，两套模式就可以并存。
