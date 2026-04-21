#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置常量和设置
"""

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# 模型配置
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B"
DEFAULT_REALTIME_QWEN_MODEL = "Qwen/Qwen3-4B"
DEFAULT_QWEN_LOCAL_DIRS = [
    "Qwen3-4B",
    "Qwen3-4B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B-Instruct",
]
DEFAULT_QWEN_LORA_DIRS = [
    "qwen3-4b-oralcoach-stage1-v3-closing-lora",
    "qwen3-4b-oralcoach-stage1-v2-6000-lora",
    "qwen3-4b-oralcoach-stage1-lora",
]
DEFAULT_SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
DEFAULT_SILERO_VAD_REPO = "snakers4/silero-vad"

# 性能参数
WHISPER_MODEL_SIZE = "base"
REALTIME_CHUNK_MS = 100
REALTIME_PARTIAL_ASR_INTERVAL_MS = 600
REALTIME_MIN_UTTERANCE_MS = 300
REALTIME_MAX_HISTORY_TURNS = 10
REALTIME_VAD_START_THRESHOLD = 0.55
REALTIME_VAD_END_THRESHOLD = 0.35
REALTIME_INTERRUPT_THRESHOLD = 0.50
REALTIME_INTERRUPT_GRACE_MS = 120
REALTIME_AUDIO_CHUNK_SAMPLES = SAMPLE_RATE * REALTIME_CHUNK_MS // 1000
REALTIME_TTS_CHUNK_MS = 240
REALTIME_TTS_AUDIO_CHUNK_SAMPLES = SAMPLE_RATE * REALTIME_TTS_CHUNK_MS // 1000

# 语言检测关键词
LEARNING_KEYWORDS = [
    '學習', '学习', 'practice', '练习', '对话', 'conversation',
    '口语', 'speaking', '英语', 'english', '场景', 'scene',
    '現在', '正在', '进行', '進行', '開始', '开始', '場景', '场景'
]

GREETING_KEYWORDS = ['你好', 'hello', 'hi', 'hey']

# 文本清理关键词
SKIP_KEYWORDS = [
    '同时注意', '好的，请', '欢迎加入', '明白了，请',
    '这是一个', '请用', '回复应该', '你的回复',
    '例如', '比如', '注意', '请继续',
    '好了', '欢迎', '加入', '讨论',
    '直接开始', '现在请', '请直接', '你的回复',
    '回复要求', '只回复', '不要', '应该',
    '让我想想', '我想想', '思考', '考虑',
    'system:', 'system', '用户说', '回复',
    '作为ai', '我是ai', '我应该', '我需要',
    '让我来', '我会', '我可以', '开始',
    '现在', '好的', '明白了', '好的'
]
