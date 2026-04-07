# 🚀 Flask后端服务使用指南

## 📋 概述

本项目已完成Flask后端API服务的开发，将现有的语音对话系统封装为RESTful API接口，供前端应用（uni-app）调用。

## 🏗️ 架构说明

### 核心组件
- **Flask应用** (`code/flask_app.py`): 主API服务器
- **音频处理器** (`code/audio_processor.py`): 语音识别功能
- **对话引擎** (`code/conversation_engine.py`): AI对话生成
- **语音合成引擎** (`code/tts_engine.py`): 文字转语音

### API接口列表

| 接口 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/api/health` | GET | 健康检查 | 检查服务是否运行正常 |
| `/api/chat` | POST | 文本对话 | 接收用户消息，返回AI回复 |
| `/api/speech-to-text` | POST | 语音识别 | 将音频转换为文字 |
| `/api/text-to-speech` | POST | 语音合成 | 将文字转换为语音 |
| `/upload` | POST | 文件上传 | 上传音频文件 |
| `/api/audio/<file_id>` | GET | 获取音频 | 下载生成的音频文件 |


## 📖 API详细说明

### 1. 文本对话接口

**请求示例：**
```json
POST /api/chat
{
  "message": "Hello, how are you?",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ],
  "mode": "normal"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "text": "I'm doing great, thank you!",
    "metadata": {
      "tokens_used": 25,
      "processing_time": 1234.56,
      "language_detected": "en"
    }
  },
  "message": "处理成功"
}
```

### 2. 语音识别接口

**请求示例：**
```json
POST /api/speech-to-text
{
  "audio": "base64编码的音频数据",
  "format": "wav",
  "language": "auto"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "text": "Hello, how are you?",
    "confidence": 0.95,
    "language": "en",
    "duration": 3.5,
    "metadata": {
      "processing_time": 2345.67,
      "model_used": "whisper-base"
    }
  },
  "message": "语音识别成功"
}
```

### 3. 语音合成接口

**请求示例：**
```json
POST /api/text-to-speech
{
  "text": "Hello! This is a test.",
  "language": "en"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "audio_url": "/api/audio/tmp_abc123.mp3",
    "duration": 2.5,
    "size": 45678,
    "metadata": {
      "processing_time": 1567.89,
      "voice_used": "en"
    }
  },
  "message": "语音合成成功"
}
```

## 🔍 错误处理

### 常见错误码

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| `INVALID_PARAMS` | 参数格式错误 | 检查请求参数 |
| `INVALID_AUDIO_FORMAT` | 音频格式不支持 | 使用wav/mp3格式 |
| `PROCESSING_FAILED` | 处理失败 | 查看服务器日志 |
| `TTS_FAILED` | 语音合成失败 | 检查网络连接 |
| `FILE_NOT_FOUND` | 文件不存在 | 确认文件ID正确 |

### 错误响应示例

```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMS",
    "message": "缺少必填参数: message"
  }
}
```



## 🛠️ 故障排查

### 问题1: 服务无法启动

**症状**: 运行flask_app.py时报错

**解决方案**:
1. 检查依赖包是否完整安装
2. 确认端口8000未被占用
3. 验证模型文件是否存在

### 问题2: 语音识别失败

**症状**: `/api/speech-to-text` 返回错误

**解决方案**:
1. 确认音频格式正确（wav/mp3）
2. 检查base64编码是否正确
3. 验证Whisper模型已加载

### 问题3: 语音合成失败

**症状**: `/api/text-to-speech` 返回错误

**解决方案**:
1. 检查网络连接（Edge TTS需要联网）
2. 确认文本内容不为空
3. 查看服务器日志获取详细错误

### 问题4: CORS跨域错误

**症状**: 前端请求被浏览器阻止

**解决方案**:
1. 确认Flask-CORS已安装
2. 检查CORS配置是否正确
3. 验证请求头设置

## 🎯 下一步

1. **前端集成**: 在uni-app中调用这些API接口
2. **功能扩展**: 根据需求添加新的API接口
3. **性能优化**: 根据实际使用情况优化响应速度
4. **部署上线**: 将服务部署到生产环境

---

**最后更新**: 2025年1月9日  
**作者**: EchoSage团队
