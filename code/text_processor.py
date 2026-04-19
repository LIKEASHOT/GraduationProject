#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理和清理功能 - 简化版本
"""

import re
from config import SKIP_KEYWORDS


class TextProcessor:
    """文本处理器 - 只做基本清理，不截断内容"""

    @staticmethod
    def clean_model_response(response, allow_long_response=False):
        """
        清理模型回复 - 简化版本,只做必要的清理,不截断内容

        Args:
            response: 原始回复文本
            allow_long_response: 是否允许长回复（未使用）
        """
        response = TextProcessor.remove_thinking(response)
        response = TextProcessor.remove_markdown_formatting(response)

        # 移除Qwen模型的特殊标记
        response = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', response, flags=re.DOTALL)
        response = re.sub(r'<\|im_start\|>.*', '', response)
        response = re.sub(r'<\|im_end\|>', '', response)

        # 移除对话标记 - 如果整行都是标记则跳过
        if '\n' in response:
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # 跳过纯标记行
                if re.match(r'^(Human|Assistant|User|System)\s*:', line, flags=re.IGNORECASE):
                    continue
                if line:
                    cleaned_lines.append(line)
            response = ' '.join(cleaned_lines)

        # 移除行内的对话标记
        response = re.sub(r'\b(Human|Assistant|User|System)\s*:\s*', '', response, flags=re.IGNORECASE)

        # 移除多余的空格
        response = re.sub(r'\s+', ' ', response).strip()

        return response if response else "Hello!"

    @staticmethod
    def clean_response(response):
        """清理回复文本 - 基本清理"""
        response = TextProcessor.remove_thinking(response)
        response = TextProcessor.remove_markdown_formatting(response)
        # 移除多余的换行和空格
        response = response.replace('\n', ' ').strip()
        # 移除重复的空格
        while '  ' in response:
            response = response.replace('  ', ' ')
        return response

    @staticmethod
    def remove_thinking(response):
        """Remove Qwen3 thinking traces before returning text to clients."""
        response = str(response or "")
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<think>.*', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'</think>', '', response, flags=re.IGNORECASE)
        return response.strip()

    @staticmethod
    def remove_markdown_formatting(response):
        """Strip lightweight Markdown markers that are awkward in chat/TTS."""
        response = str(response or "")
        response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
        response = re.sub(r"__(.*?)__", r"\1", response)
        response = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", response)
        response = re.sub(r"`([^`]+)`", r"\1", response)
        response = re.sub(r"(?m)^\s*[-*]\s+", "", response)
        response = re.sub(r"(?m)^\s*#{1,6}\s*", "", response)
        return response.strip()
