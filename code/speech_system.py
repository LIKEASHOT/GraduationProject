#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音对话系统主类
"""

import os
import time
import torch
from datetime import datetime
import warnings
import argparse

# 设置控制台编码为UTF-8（Windows兼容）
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

warnings.filterwarnings("ignore")

from config import DEFAULT_QWEN_MODEL
from audio_processor import AudioProcessor
from conversation_engine import ConversationEngine
from tts_engine import TTSEngine


class CompleteSpeechSystem:
    """完整的语音对话系统"""

    def __init__(self, qwen_model_name=DEFAULT_QWEN_MODEL, local_model_path=None, lora_adapter_path=None, prefer_local_tts=False):
        """
        初始化完整的语音对话系统

        Args:
            qwen_model_name: Qwen模型名称（在线下载）
            local_model_path: 本地模型路径（离线使用）
            prefer_local_tts: 是否优先使用本地TTS模型
        """
        self.qwen_model_name = qwen_model_name
        self.local_model_path = local_model_path
        self.lora_adapter_path = lora_adapter_path

        # 性能计时器
        self.timing = {
            'init_start': time.time(),
            'whisper_init': 0,
            'qwen_init': 0,
            'tts_init': 0,
            'audio_load': 0,
            'speech_recognition': 0,
            'response_generation': 0,
            'text_to_speech': 0,
            'total': 0
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 初始化各个组件
        self.audio_processor = AudioProcessor()
        self.conversation_engine = ConversationEngine()
        self.tts_engine = TTSEngine(prefer_edge_tts=True, prefer_local_tts=True)  # Prefer local MOSS-TTS-Nano, then Edge/gTTS.

        self._init_components()

        # 计算初始化总时间
        init_total_time = time.time() - self.timing['init_start']
        self.timing['total'] = init_total_time
        print(f"⏱️  初始化总耗时: {init_total_time:.2f}秒")
        print("⏱️  性能统计:")
        print(f"  📝 Whisper加载: {self.timing['whisper_init']:.2f}秒")
        print(f"  🤖 Qwen加载: {self.timing['qwen_init']:.2f}秒")
        print(f"  🔊 TTS加载: {self.timing['tts_init']:.2f}秒")

    def _init_components(self):
        """初始化各个组件"""
        # 初始化Whisper
        start_time = time.time()
        self.audio_processor.init_whisper()
        self.timing['whisper_init'] = time.time() - start_time
        print(f"✅ Whisper模型加载成功! (耗时: {self.timing['whisper_init']:.2f}秒)")

        # 初始化Qwen模型
        start_time = time.time()
        if not self.conversation_engine.init_model(
            self.qwen_model_name,
            self.local_model_path,
            self.lora_adapter_path
        ):
            print("模型初始化失败，将使用简单回复逻辑")
        self.timing['qwen_init'] = time.time() - start_time

        # 初始化TTS
        start_time = time.time()
        self.tts_engine.init_tts()
        self.timing['tts_init'] = time.time() - start_time
        print(f"✅ 语音合成引擎初始化成功! (耗时: {self.timing['tts_init']:.2f}秒)")

    def run_conversation(self, audio_file=None):
        """
        运行完整的对话流程

        Args:
            audio_file: 音频文件路径，如果为None则进行实时录制
        """
        print("=" * 60)
        if audio_file:
            print("语音识别文件模式")
            print(f"处理文件: {audio_file}")
        else:
            print("完整的语音对话系统启动")
            print("支持中英文语音输入 -> AI对话 -> 英文语音输出")
            print("按回车键开始录制，按Ctrl+C退出")
        print("=" * 60)

        try:
            if audio_file:
                # 文件模式：只处理一次
                self._process_audio_file(audio_file)
            else:
                # 实时录制模式
                self._run_real_time_conversation()

        except KeyboardInterrupt:
            print("\n系统退出，感谢使用！")

    def _process_audio_file(self, audio_file):
        """处理音频文件"""
        start_time = time.time()

        # 1. 加载音频
        audio_data = self.audio_processor.load_audio_from_file(audio_file)
        if audio_data is None:
            return

        self.timing['audio_load'] = time.time() - start_time

        # 2. 语音转文字
        start_time = time.time()
        recognized_text = self.audio_processor.speech_to_text(audio_data)
        self.timing['speech_recognition'] = time.time() - start_time

        if not recognized_text.strip():
            print("未识别到有效语音")
            return

        # 3. 生成回复（文件模式允许中等长度回复，便于用户阅读）
        start_time = time.time()
        response_text = self.conversation_engine.generate_response(recognized_text, medium_response=True)
        self.timing['response_generation'] = time.time() - start_time

        # 4. 文字转语音（文件模式：同步播放，确保语音完整播放）
        start_time = time.time()
        self.tts_engine.text_to_speech(response_text, async_play=False)
        # 记录完整播放时间
        self.timing['text_to_speech'] = time.time() - start_time

        # 显示处理时间统计
        print("-" * 60)
        print("📊 详细时间统计:")

        # 处理阶段时间
        print(f"  📁 音频文件加载: {self.timing['audio_load']:.3f}秒")
        print(f"  🎙️  语音识别: {self.timing['speech_recognition']:.3f}秒")
        print(f"  🤖 AI回复生成: {self.timing['response_generation']:.3f}秒")
        print(f"  🔊 语音合成: {self.timing['text_to_speech']:.3f}秒")

        processing_total = (
            self.timing['audio_load'] +
            self.timing['speech_recognition'] +
            self.timing['response_generation'] +
            self.timing['text_to_speech']
        )
        print(f"  ⏱️  处理阶段总计: {processing_total:.3f}秒")

        # 系统初始化时间（从类初始化开始到处理开始）
        init_total = (
            self.timing['whisper_init'] +
            self.timing['qwen_init'] +
            self.timing['tts_init']
        )
        print(f"  🔧 系统初始化: {init_total:.3f}秒")

        # 完整流程总时间
        total_time = init_total + processing_total
        print(f"  🎯 完整流程总计: {total_time:.3f}秒")

        print("✅ 文件处理完成！")

    def _run_real_time_conversation(self):
        """运行实时对话"""
        while True:
            input("准备好后按回车开始录制...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"input_{timestamp}.wav"

            # 1. 录制语音
            audio_data = self.audio_processor.record_audio(duration=5, save_path=audio_file)

            # 2. 语音转文字
            recognized_text = self.audio_processor.speech_to_text(audio_data)

            if not recognized_text.strip():
                print("未识别到有效语音，请重试")
                continue

            # 3. 生成回复
            response_text = self.conversation_engine.generate_response(recognized_text)

            # 4. 文字转语音
            self.tts_engine.text_to_speech(response_text)

            print("-" * 60)

    # 代理方法，保持向后兼容性
    def record_audio(self, *args, **kwargs):
        return self.audio_processor.record_audio(*args, **kwargs)

    def load_audio_from_file(self, *args, **kwargs):
        return self.audio_processor.load_audio_from_file(*args, **kwargs)

    def speech_to_text(self, *args, **kwargs):
        return self.audio_processor.speech_to_text(*args, **kwargs)

    def generate_response(self, *args, **kwargs):
        return self.conversation_engine.generate_response(*args, **kwargs)

    def text_to_speech(self, *args, **kwargs):
        return self.tts_engine.text_to_speech(*args, **kwargs)
