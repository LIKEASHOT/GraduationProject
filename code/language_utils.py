#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语言检测和处理工具
"""

from config import LEARNING_KEYWORDS, GREETING_KEYWORDS


class LanguageUtils:
    """语言工具类"""

    @staticmethod
    def detect_text_language(text):
        """
        检测文本语言

        Args:
            text: 输入文本

        Returns:
            str: 'zh' for Chinese, 'en' for English, 'mixed' for mixed
        """
        chinese_chars = 0
        english_chars = 0

        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                chinese_chars += 1
            elif char.isascii() and char.isalpha():
                english_chars += 1

        total_chars = chinese_chars + english_chars

        if total_chars == 0:
            return 'en'  # 默认英文

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.7:
            return 'zh'
        elif english_ratio > 0.7:
            return 'en'
        else:
            return 'mixed'

    @staticmethod
    def detect_conversation_intent(text_input):
        """
        检测对话意图

        Args:
            text_input: 用户输入文本

        Returns:
            dict: 意图信息
        """
        text_lower = text_input.lower()

        # 检测学习/对话场景
        is_learning_scenario = any(keyword in text_lower for keyword in LEARNING_KEYWORDS)

        # 检测简单问候
        is_simple_greeting = any(keyword in text_lower for keyword in GREETING_KEYWORDS) and len(text_input.split()) < 5

        return {
            'is_learning_scenario': is_learning_scenario,
            'is_simple_greeting': is_simple_greeting,
            'text_lower': text_lower
        }
