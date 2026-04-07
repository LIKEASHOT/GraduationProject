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
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 性能参数
WHISPER_MODEL_SIZE = "base"

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
