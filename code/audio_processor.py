#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块
负责音频录制、加载和语音识别功能
"""

import os
import sys
import numpy as np
import pyaudio
import soundfile as sf
import whisper
from datetime import datetime

from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, WHISPER_MODEL_SIZE


class AudioProcessor:
    """音频处理器"""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        self.whisper_model = None

    def init_whisper(self):
        """初始化Whisper模型"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        return self.whisper_model

    def record_audio(self, duration=5, save_path=None):
        """
        录制音频

        Args:
            duration: 录制时长(秒)
            save_path: 保存路径

        Returns:
            audio_data: numpy数组格式的音频数据
        """
        print(f"开始录制音频，时长{duration}秒...请说话...")

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        for i in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        print("录制完成！")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # 转换为numpy数组
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        if save_path:
            sf.write(save_path, audio_data, self.sample_rate)
            print(f"音频已保存到: {save_path}")

        return audio_data

    def load_audio_from_file(self, file_path):
        """
        从文件加载音频 - 超快版本

        Args:
            file_path: 音频文件路径

        Returns:
            audio_data: numpy数组格式的音频数据
        """
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None

        try:
            # 打印详细信息用于调试
            file_size = os.path.getsize(file_path)
            print(f"📂 正在加载音频: {file_path}")
            print(f"📊 文件大小: {file_size} bytes")

            # 方法1: 尝试使用 soundfile 加载
            try:
                audio_data, sample_rate = sf.read(file_path, dtype='float32')
                print(f"✅ 使用 soundfile 加载成功")
            except Exception as sf_error:
                print(f"⚠️ soundfile 加载失败: {sf_error}")
                print(f"🔄 尝试使用 wave 库加载...")

                # 方法2: 使用标准库 wave 加载 WAV 文件
                import wave
                with wave.open(file_path, 'rb') as wav_file:
                    # 获取音频参数
                    channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    print(f"📋 WAV 参数: 采样率={sample_rate}Hz, 声道={channels}, 位深={sampwidth*8}bit")

                    # 读取音频数据
                    audio_bytes = wav_file.readframes(n_frames)

                    # 转换为 numpy 数组
                    if sampwidth == 1:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                        audio_data = (audio_data - 128) / 128.0  # 转换到 [-1, 1]
                    elif sampwidth == 2:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_data = audio_data / 32768.0  # 转换到 [-1, 1]
                    elif sampwidth == 4:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                        audio_data = audio_data / 2147483648.0  # 转换到 [-1, 1]
                    else:
                        raise ValueError(f"不支持的位深: {sampwidth*8}bit")

                    # 如果是立体声，转换为单声道
                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                        print(f"🔄 立体声转单声道")
                    elif channels > 2:
                        audio_data = audio_data.reshape(-1, channels).mean(axis=1)
                        print(f"🔄 多声道转单声道 ({channels} -> 1)")

                    audio_data = audio_data.astype('float32')
                    print(f"✅ 使用 wave 库加载成功")

            # 快速采样率转换（如果需要）
            if sample_rate != self.sample_rate:
                from scipy import signal
                target_length = int(len(audio_data) * self.sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, target_length)
                print(f"🔄 采样率转换: {sample_rate}Hz -> {self.sample_rate}Hz")

            duration = len(audio_data) / self.sample_rate
            print(f"✅ 音频文件加载完成 (时长: {duration:.1f}秒)")
            return audio_data

        except Exception as e:
            import traceback
            print(f"❌ 音频加载失败: {str(e)}")
            print(f"📋 完整错误信息:")
            traceback.print_exc()
            return None

    def speech_to_text(self, audio_data):
        """
        语音转文字 - 支持中英文混合输入

        Args:
            audio_data: numpy数组格式的音频数据

        Returns:
            text: 识别出的文字
        """
        print("正在进行语音识别...")
        print(f"📊 音频数据: 长度={len(audio_data)}, dtype={audio_data.dtype}, 范围=[{audio_data.min():.3f}, {audio_data.max():.3f}]")

        if self.whisper_model is None:
            print("❌ Whisper模型未初始化")
            return ""

        # 检查音频是否有效
        if len(audio_data) == 0:
            print("❌ 音频数据为空")
            return ""

        # 检查音频音量
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"📊 音频RMS音量: {rms:.6f}")

        if rms < 0.001:
            print("⚠️ 音频音量过低，可能是静音")

        try:
            # 使用Whisper进行语音识别 - 自动检测语言，支持混合输入
            result = self.whisper_model.transcribe(
                audio_data,
                language=None,  # 自动检测语言
                task='transcribe',  # 转录任务
                fp16=False,  # CPU模式禁用半精度
                beam_size=1,  # 减小束搜索，提高速度
                patience=1.0,  # 标准耐心值
                temperature=0,  # 贪婪解码，速度最快
                best_of=1,  # 只使用最好的结果
                verbose=True  # 启用详细输出用于调试
            )

            # 打印完整的识别结果用于调试
            print(f"🔍 Whisper完整结果: {result}")

            text = result["text"].strip()

            # 检测识别结果的语言
            detected_lang = self._detect_text_language(text)
            print(f"识别结果: '{text}' (检测语言: {detected_lang})")

            return text

        except Exception as e:
            import traceback
            print(f"❌ 语音识别失败: {e}")
            traceback.print_exc()
            return ""

    def _detect_text_language(self, text):
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
