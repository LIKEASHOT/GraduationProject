#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio processing utilities.
Provides audio recording, file loading, and SenseVoice-based ASR.
"""

import os
import re
import wave
from datetime import datetime

import numpy as np
import pyaudio
import soundfile as sf

from config import CHANNELS, CHUNK_SIZE, DEFAULT_SENSEVOICE_MODEL, SAMPLE_RATE


class AudioProcessor:
    """Audio processing helper centered on SenseVoice ASR."""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        self.asr_model = None
        self.asr_backend_name = "sensevoice"
        self.local_model_path = os.environ.get("SENSEVOICE_MODEL_PATH")

    def init_sensevoice(self):
        """Initialize the SenseVoice model."""
        if self.asr_model is not None:
            return self.asr_model

        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "SenseVoice ASR requires `funasr`. Please install it and configure "
                "`SENSEVOICE_MODEL_PATH` if you use a local model."
            ) from exc

        candidate_paths = []
        if self.local_model_path:
            candidate_paths.append(self.local_model_path)
        candidate_paths.append(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "SenseVoiceSmall")
        )

        model_target = DEFAULT_SENSEVOICE_MODEL
        for path in candidate_paths:
            if path and os.path.exists(path):
                model_target = path
                break

        try:
            self.asr_model = AutoModel(model=model_target, vad_model=None, punc_model=None, spk_model=None)
            print(f"SenseVoice ASR ready: {model_target}")
            return self.asr_model
        except Exception as exc:
            raise RuntimeError(f"SenseVoice model initialization failed: {model_target}") from exc

    def init_whisper(self):
        """Backward-compatible alias. The project now uses SenseVoice."""
        return self.init_sensevoice()

    def record_audio(self, duration=5, save_path=None):
        """Record audio from the default microphone."""
        print(f"Start recording for {duration} seconds...")

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        if save_path:
            sf.write(save_path, audio_data, self.sample_rate)
            print(f"Audio saved to: {save_path}")

        return audio_data

    def load_audio_from_file(self, file_path):
        """Load an audio file and convert it to mono float32 PCM."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        try:
            file_size = os.path.getsize(file_path)
            print(f"Loading audio: {file_path}")
            print(f"File size: {file_size} bytes")

            try:
                audio_data, sample_rate = sf.read(file_path, dtype="float32")
                print("Loaded audio with soundfile")
            except Exception as sf_error:
                print(f"soundfile load failed: {sf_error}")
                print("Trying wave loader...")

                with wave.open(file_path, "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_bytes = wav_file.readframes(n_frames)

                    if sampwidth == 1:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                        audio_data = (audio_data - 128) / 128.0
                    elif sampwidth == 2:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16) / 32768.0
                    elif sampwidth == 4:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth * 8}bit")

                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                    elif channels > 2:
                        audio_data = audio_data.reshape(-1, channels).mean(axis=1)

                    audio_data = audio_data.astype("float32")
                    print("Loaded audio with wave")

            if getattr(audio_data, "ndim", 1) > 1:
                audio_data = audio_data.mean(axis=1)

            if sample_rate != self.sample_rate:
                from scipy import signal

                target_length = int(len(audio_data) * self.sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, target_length)
                print(f"Resampled audio: {sample_rate}Hz -> {self.sample_rate}Hz")

            duration = len(audio_data) / self.sample_rate
            print(f"Audio load finished. Duration: {duration:.1f}s")
            return np.asarray(audio_data, dtype="float32")

        except Exception as exc:
            import traceback

            print(f"Audio load failed: {exc}")
            traceback.print_exc()
            return None

    def speech_to_text(self, audio_data):
        """Run SenseVoice speech recognition."""
        print("Running speech recognition...")

        audio_data = np.asarray(audio_data, dtype=np.float32).flatten()
        if audio_data.size == 0:
            print("Empty audio input")
            return ""

        if self.asr_model is None:
            print("SenseVoice model is not initialized")
            return ""

        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        print(
            f"Audio stats: length={len(audio_data)}, dtype={audio_data.dtype}, "
            f"range=[{audio_data.min():.3f}, {audio_data.max():.3f}], rms={rms:.6f}"
        )

        try:
            result = self.asr_model.generate(
                input=audio_data,
                cache={},
                is_final=True,
                language="auto",
            )
            print(f"SenseVoice raw result: {result}")

            if isinstance(result, list) and result:
                text = self._clean_sensevoice_text(str(result[0].get("text", "")).strip())
            elif isinstance(result, dict):
                text = self._clean_sensevoice_text(str(result.get("text", "")).strip())
            else:
                text = ""

            detected_lang = self._detect_text_language(text)
            print(f"Recognized text: '{text}' (language: {detected_lang})")
            return text

        except Exception as exc:
            import traceback

            print(f"SenseVoice speech recognition failed: {exc}")
            traceback.print_exc()
            return ""

    def _detect_text_language(self, text):
        """Detect whether text is mostly Chinese, English, or mixed."""
        chinese_chars = 0
        english_chars = 0

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                chinese_chars += 1
            elif char.isascii() and char.isalpha():
                english_chars += 1

        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.7:
            return "zh"
        if english_ratio > 0.7:
            return "en"
        return "mixed"

    @staticmethod
    def _clean_sensevoice_text(text):
        """Remove SenseVoice inline control tokens."""
        return re.sub(r"<\|[^|]+?\|>", "", text).strip()
