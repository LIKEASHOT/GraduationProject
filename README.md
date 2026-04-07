<p align="center">
  <img src="logo.png" alt="EchoSage Logo" width="200"/>
</p>

<h1 align="center">🎙️ EchoSage — 智能英语语音对话学习应用</h1>

<p align="center">
  <strong>基于端到端语音对话模型的英语口语学习平台</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Qwen2.5-7B-6366F1?style=flat-square" alt="Qwen"/>
  <img src="https://img.shields.io/badge/Whisper-Base-74AA9C?style=flat-square&logo=openai&logoColor=white" alt="Whisper"/>
  <img src="https://img.shields.io/badge/uni--app-Vue3-4FC08D?style=flat-square&logo=vuedotjs&logoColor=white" alt="uni-app"/>
</p>

---

## 📋 项目简介

**EchoSage** 是一款面向非英语母语者的智能英语语音对话学习应用。系统采用端到端架构，集成 **语音识别（ASR）**、**大语言模型对话（LLM）** 和 **语音合成（TTS）** 三大核心模块，为用户提供沉浸式的英语口语练习体验。

### ✨ 核心特性

- 🗣️ **智能语音对话** — 支持中英文混合语音输入，AI 智能回复
- 📞 **多模式学习** — 普通对话模式 & 电话模拟模式，覆盖多种学习场景
- 🧠 **大模型驱动** — 基于 Qwen2.5 大语言模型，生成自然流畅的英语回复
- 🎧 **高质量语音** — 集成 Edge TTS / gTTS 多引擎语音合成
- 📱 **跨平台前端** — 基于 uni-app 框架，支持多端部署
- ⚡ **低延迟响应** — 优化的 Prompt 工程 + 精简推理，实现快速对话交互

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      EchoSage 系统架构                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    RESTful API    ┌────────────────────┐  │
│  │   Frontend   │ ◄──────────────► │     Backend        │  │
│  │  (uni-app)   │   HTTP / JSON    │   (Flask Server)   │  │
│  └──────────────┘                  └────────┬───────────┘  │
│                                             │              │
│                              ┌──────────────┼──────────┐   │
│                              │              │          │   │
│                         ┌────▼────┐  ┌──────▼───┐ ┌───▼─┐ │
│                         │  ASR    │  │   LLM    │ │ TTS │ │
│                         │ Whisper │  │ Qwen2.5  │ │Edge │ │
│                         └─────────┘  └──────────┘ └─────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
GraduationProject/
├── 📂 frontend/                         # 前端应用 (uni-app + Vue3)
│   ├── pages/                           # 页面目录
│   │   ├── index/                       # 首页（模式选择）
│   │   └── chat/                        # 对话页面
│   ├── static/                          # 静态资源
│   └── ...
│
├── 📂 backend/                          # 后端服务 (Python + Flask)
│   ├── flask_app.py                     # Flask API 主服务器
│   ├── audio_processor.py               # 音频处理 & 语音识别模块
│   ├── conversation_engine.py           # 对话引擎（Qwen2.5 推理）
│   ├── tts_engine.py                    # 语音合成引擎
│   ├── text_processor.py               # 文本后处理模块
│   ├── speech_system.py                # 语音系统主控制器
│   ├── language_utils.py               # 语言检测工具
│   ├── config.py                        # 配置文件
│   ├── main.py                          # CLI 入口文件
│   ├── download.py                      # 模型下载脚本
│   ├── requirements.txt                 # Python 依赖包
│   ├── start_flask_server.ps1           # 服务启动脚本
│   ├── API_DOCUMENTATION.md             # API 接口文档
│   ├── FLASK_BACKEND_README.md          # 后端使用指南
│   ├── OPTIMIZATION_GUIDE.md            # 模型优化指南
│   └── README.md                        # 后端说明文档
│
├── logo.png                             # 项目 Logo
└── README.md                            # 项目总览（本文件）
```

---

## ⚙️ 技术栈

| 层级 | 技术 | 说明 |
|:---:|:---|:---|
| **前端** | uni-app + Vue3 | 跨平台移动端框架 |
| **后端** | Flask 3.0 + Flask-CORS | 轻量级 RESTful API 服务 |
| **语音识别** | OpenAI Whisper (base) | 高精度中英文语音转文字 |
| **对话模型** | Qwen2.5-7B-Instruct | 通义千问大语言模型 |
| **语音合成** | Edge TTS / gTTS | 多引擎高质量文字转语音 |
| **深度学习** | PyTorch 2.0+ | 模型推理引擎 |
| **音频处理** | scipy / soundfile / pygame | 音频 I/O 与播放 |

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- conda（推荐）或 pip
- Node.js 16+（前端开发）
- CUDA（可选，GPU 加速推荐）

### 1. 克隆仓库

```bash
git clone https://github.com/LIKEASHOT/GraduationProject.git
cd GraduationProject
```

### 2. 后端部署

```bash
cd backend

# 创建并激活 conda 环境
conda create -n echosage python=3.11 -y
conda activate echosage

# 安装依赖（国内用户推荐使用镜像源）
pip install -r requirements.txt
# 或使用清华源加速
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt

# 启动 Flask 后端服务
python flask_app.py
```

服务启动后，API 将在 `http://localhost:8000` 上运行。

### 3. 前端部署

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

---

## 📡 API 接口概览

后端提供以下 RESTful API 接口：

| 接口路径 | 方法 | 功能描述 |
|:---|:---:|:---|
| `/api/health` | `GET` | 服务健康检查 |
| `/api/chat` | `POST` | 文本对话 — 发送消息获取 AI 回复 |
| `/api/speech-to-text` | `POST` | 语音识别 — 音频转文字 |
| `/api/text-to-speech` | `POST` | 语音合成 — 文字转音频 |
| `/upload` | `POST` | 音频文件上传 |
| `/api/audio/<file_id>` | `GET` | 获取生成的音频文件 |

### 示例请求

```bash
# 文本对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "mode": "normal"}'

# 语音识别
curl -X POST http://localhost:8000/api/speech-to-text \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64_encoded_audio>", "format": "wav"}'

# 语音合成
curl -X POST http://localhost:8000/api/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Nice to meet you.", "language": "en"}'
```

> 📖 完整 API 文档请参阅 [`backend/API_DOCUMENTATION.md`](backend/API_DOCUMENTATION.md)

---

## 📊 性能指标

| 指标 | 数值 |
|:---|:---:|
| 文本对话响应 | < 3 秒 |
| 语音识别延迟 | < 5 秒 |
| 语音合成延迟 | < 2 秒 |
| AI 推理时间 | ~1.5-2 秒 |
| 支持语言 | 中文 / 英文 / 混合 |

---

## 🧩 核心模块说明

### 🎤 语音识别模块 (`audio_processor.py`)
基于 OpenAI Whisper 模型，支持中英文混合语音识别，自动检测语言类型，提供高精度的音频转文字能力。

### 🧠 对话引擎 (`conversation_engine.py`)
使用 Qwen2.5 大语言模型，通过优化的 Prompt 工程实现自然、简洁的对话生成。支持多轮对话上下文、学习场景识别以及智能参数调节。

### 🔊 语音合成引擎 (`tts_engine.py`)
集成 Edge TTS 和 gTTS 双引擎，支持中英文高质量语音合成。自动选择最佳引擎，提供流畅自然的语音输出。

### 📝 文本处理模块 (`text_processor.py`)
负责 AI 回复的后处理，包括内容清理、格式标准化和语言检测，确保输出文本的质量和一致性。

---

## 📚 相关文档

| 文档 | 说明 |
|:---|:---|
| [`backend/API_DOCUMENTATION.md`](backend/API_DOCUMENTATION.md) | API 接口完整文档 |
| [`backend/FLASK_BACKEND_README.md`](backend/FLASK_BACKEND_README.md) | 后端服务使用指南 |
| [`backend/OPTIMIZATION_GUIDE.md`](backend/OPTIMIZATION_GUIDE.md) | 模型优化 & Prompt 工程指南 |

---

## 🛠️ 开发指南

### 项目构建

```bash
# 后端：启动开发服务（支持热重载）
cd backend && python flask_app.py

# 前端：启动开发服务
cd frontend && npm run dev
```

### 目录约定

- **前端代码** → `frontend/`
- **后端代码** → `backend/`
- **模型文件** → 本地存放，不上传至仓库（已在 `.gitignore` 中排除）

---

## 📄 License

本项目仅用于学术研究和毕业设计，暂不提供开源许可证。

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/LIKEASHOT">LIKEASHOT</a> · EchoSage Team</sub>
</p>
