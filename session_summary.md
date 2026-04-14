# Echo 全双工语音界面 - 开发进度总结

> **项目路径**: `e:\Documents\HBuilderProjects\echo`
> **核心文件**: `pages/chat/components/FullDuplexScreen.vue`
> **日期**: 2026-04-14

---

## 一、架构概览

```
┌─────────────────────────────────────────────────┐
│  chat.vue (逻辑层 - 消息持久化、模式切换)         │
│  ├─ messages ref (主消息列表)                     │
│  ├─ saveHistory() → localStorage                 │
│  └─ handleDuplexMessages() ← sync-messages 事件  │
└─────────────┬───────────────────────────────────┘
              │ v-if="currentMode === 'phone'"
              │ :messages="messages"
              │ @end-call / @sync-messages
┌─────────────▼───────────────────────────────────┐
│  FullDuplexScreen.vue                            │
│                                                  │
│  <script>  (Options API - 模块级 ref + methods)  │
│  ├─ _localMessages, _connStatus 等 (模块级 ref)  │
│  └─ methods: onSubtitle, onConnStatus...         │
│     (renderjs callMethod 只认 Options API)       │
│                                                  │
│  <script setup>  (Composition API - 模板绑定)    │
│  ├─ localMessages = _localMessages (别名)        │
│  ├─ computed: statusText, dotClass, barH         │
│  ├─ syncMessages() → emit('sync-messages')       │
│  └─ onMounted: 重置状态 + 设置 wsUrl             │
│                                                  │
│  <script module="engine" lang="renderjs">        │
│  (WebView 视图层 - 直接操作浏览器 API)            │
│  ├─ WebSocket 连接                               │
│  ├─ Silero VAD (ort-wasm + vad-web)              │
│  ├─ 能量 VAD 兜底                                │
│  ├─ TTS 音频播放队列                              │
│  └─ cachedOwner + safeCall() 跨层通信            │
└──────────────────────────────────────────────────┘
```

---

## 二、已解决的问题清单

### 1. Silero VAD 在 Android WebView 中初始化
| 问题 | 解决方案 |
|------|----------|
| `fetch()` 不支持 `file://` | monkey-patch `window.fetch`，用 XHR 替代 |
| `import()` 不支持 `file://` 的 `.mjs` | XHR 加载 mjs 文本 → Blob URL |
| `.mjs` 内 `new URL(..., import.meta.url)` 崩溃 | 字符串替换移除该死代码 |
| WASM 文件无法通过 fetch 加载 | XHR 加载为 ArrayBuffer → `ort.env.wasm.wasmBinary` 直注 |
| `AudioWorklet.addModule()` 读取 file:// 崩溃 | `processorType: 'ScriptProcessor'` 降级 |
| `Array.prototype.at` 不存在 (低版本 WebView) | 顶层 polyfill (Array + TypedArray) |

### 2. uni-app renderjs ↔ 逻辑层通信
| 问题 | 解决方案 |
|------|----------|
| `defineExpose` 的函数 renderjs `callMethod` 找不到 | 改用 Options API `methods`，模块级 ref 桥接 |
| 页面销毁后异步回调报 `nodeId of null` | `safeCall()` 包装，try-catch 吞掉 |
| `v-if` 重建后 `$ownerInstance` 指向旧实例 | `cachedOwner` 在 `onWsUrlChange` 时刷新 |

### 3. WebSocket 生命周期
| 问题 | 解决方案 |
|------|----------|
| `ws.onopen` 时桥接未就绪，状态卡"连接中" | `handleServerMsg` 入口兜底设 `connected` |
| `cleanup()` 后旧 ws.onclose 异步把新 ws 清空 | cleanup 先摘回调再 close |
| 切换模式再回来后 ws 残留不重连 | `onWsUrlChange` 入口先 `cleanup()` |

### 4. VAD 行为
| 问题 | 解决方案 |
|------|----------|
| `onVADMisfire` 未处理，isSpeaking 卡 true | 添加 `onVADMisfire` 回调重置状态 |

### 5. 消息持久化
| 问题 | 解决方案 |
|------|----------|
| 通话消息不保存到历史 | 新增 `syncMessages()` + `emit('sync-messages')` |
| chat.vue 接收同步消息 | 新增 `handleDuplexMessages()` 合并 + 立即 `saveHistory()` |

### 6. 模块级状态残留
| 问题 | 解决方案 |
|------|----------|
| `_connStatus` 等跨实例残留 | `onMounted` 时重置所有模块级 ref |

---

## 三、当前状态 (截至 2026-04-14 15:40)

### ✅ 已确认工作
- Silero VAD 初始化成功 + 语音检测正常
- WebSocket 连接 + 音频发送 + 后端接收
- TTS 音频播放
- 后端 `user_text` / `ai_text` 事件发送

### ⚠️ 待验证（刚修完，还没测试结果）
1. **v-if 重建后 UI 更新** — `cachedOwner` 方案是否让 callMethod 正确到达新实例
2. **消息顺序** — 用户报告"消息倒序"，可能是 scrollTop 没滚到底，也可能是消息数组顺序问题
3. **消息持久化** — syncMessages 是否正确同步 + saveHistory 是否正确保存

### 🔧 可能需要后续调整
- VAD 灵敏度参数微调 (`positiveSpeechThreshold: 0.6`, `minSpeechFrames: 3`)
- 能量 VAD 兜底的阈值 (`SPEECH_THRESH: 0.018`, `SILENCE_MS: 500`)
- 通话期间如果 WS 断连的自动重连逻辑

---

## 四、关键代码位置

| 功能 | 文件 | 行号区间 (约) |
|------|------|--------------|
| Options API methods (renderjs回调) | FullDuplexScreen.vue | `<script>` 块 L67-139 |
| 模块级 ref 声明 | FullDuplexScreen.vue | L76-82 |
| script setup (模板绑定) | FullDuplexScreen.vue | `<script setup>` 块 L141-230 |
| syncMessages / endCall | FullDuplexScreen.vue | L195-218 |
| renderjs safeCall + cachedOwner | FullDuplexScreen.vue | renderjs 块 L284-296 |
| onWsUrlChange (入口) | FullDuplexScreen.vue | renderjs 块 L298-306 |
| cleanup (WS 竞态修复) | FullDuplexScreen.vue | renderjs 块 L786-798 |
| tryLoadSileroVAD | FullDuplexScreen.vue | renderjs 块 L397-517 |
| handleDuplexMessages | chat.vue | L157-174 |
| saveHistory | chat.vue | L242-328 |
| switchMode | chat.vue | L176-243 |

---

## 五、后端协议 (WebSocket 事件格式)

### 前端 → 后端
```json
{ "event": "session_start" }
{ "event": "audio_stream", "payload": { "audio_chunk": "<base64>", "sample_rate": 16000, "format": "pcm", "is_complete": true } }
{ "event": "speech_end" }
```

### 后端 → 前端
```json
{ "event": "user_text", "payload": { "turn_id": "turn_0001", "message_id": "msg_0001_user", "text": "你是谁", "is_final": true } }
{ "event": "ai_text", "payload": { "turn_id": "turn_0001", "message_id": "msg_0001", "text": "完整回复文本", "is_final": false } }
{ "event": "ai_text", "payload": { "turn_id": "turn_0001", "message_id": "msg_0001", "text": "完整回复文本", "is_final": true } }
{ "event": "tts_audio_chunk", "payload": { "audio": "<base64 PCM>" } }
{ "event": "interrupt" }
{ "event": "turn_end" }
```

> **注意**: `ai_text` 的 `text` 字段是**完整文本**，不是增量 delta。前端用 `=` 覆盖而非 `+=` 追加。

---

## 六、静态资源清单 (`static/vad/`)

| 文件 | 用途 |
|------|------|
| `ort.min.js` | ONNX Runtime Web |
| `bundle.min.js` | vad-web 打包 |
| `ort-wasm-simd-threaded.mjs` | ORT WASM 加载器 (被转为 blob URL) |
| `ort-wasm-simd-threaded.wasm` | ORT WASM 二进制 (被直注为 ArrayBuffer) |
| `silero_vad_v5.onnx` | Silero VAD v5 模型 (fetch patch 加载) |
