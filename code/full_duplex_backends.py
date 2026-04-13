#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model adapters for the realtime pseudo full-duplex backend.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

import numpy as np
from config import (
    DEFAULT_COSYVOICE_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_REALTIME_QWEN_MODEL,
    DEFAULT_SENSEVOICE_MODEL,
    REALTIME_AUDIO_CHUNK_SAMPLES,
    SAMPLE_RATE,
)
from conversation_engine import ConversationEngine
from text_processor import TextProcessor
from tts_engine import TTSEngine


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_model_path(name: str) -> str:
    return os.path.join(_project_root(), "models", name)


def split_sentences(text: str) -> List[str]:
    normalized = TextProcessor.clean_response(text)
    if not normalized:
        return []

    sentence_breaks = "\u3002\uff01\uff1f!?."
    soft_breaks = "\uff0c,\uff1b;:\uff1a"
    pieces = re.split(rf"(?<=[{sentence_breaks}{soft_breaks}])", normalized)
    segments: List[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current}{piece}".strip()
        if len(candidate) < 10 and candidate[-1] not in sentence_breaks + soft_breaks:
            current = f"{candidate} "
            continue
        if current:
            segments.append(candidate.strip())
            current = ""
        else:
            segments.append(piece)

    if current.strip():
        segments.append(current.strip())

    short_segments: List[str] = []
    for segment in segments or [normalized]:
        short_segments.extend(_split_long_tts_segment(segment, max_chars=18))
    return short_segments


def _split_long_tts_segment(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


@dataclass
class SynthesizedSegment:
    text: str
    pcm_bytes: bytes
    sample_rate: int
    audio_format: str = "pcm"

    def iter_base64_chunks(self, chunk_samples: int = REALTIME_AUDIO_CHUNK_SAMPLES) -> Iterable[str]:
        frame_size = max(1, chunk_samples) * 2
        for index in range(0, len(self.pcm_bytes), frame_size):
            yield base64.b64encode(self.pcm_bytes[index:index + frame_size]).decode("utf-8")


class SenseVoiceASRBackend:
    """ASR adapter that requires SenseVoice for realtime sessions."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, model_name: str = DEFAULT_SENSEVOICE_MODEL) -> None:
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.local_model_path = os.environ.get("SENSEVOICE_MODEL_PATH") or _default_model_path("SenseVoiceSmall")
        self.backend_name = "sensevoice"
        self._funasr_model = None

    def load(self) -> None:
        if self._funasr_model is not None:
            return

        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "Realtime ASR requires SenseVoice, but `funasr` is not installed. "
                "Install `funasr` and configure SENSEVOICE_MODEL_PATH."
            ) from exc

        model_path = self.local_model_path if os.path.exists(self.local_model_path) else self.model_name
        try:
            self._funasr_model = AutoModel(model=model_path, vad_model=None, punc_model=None, spk_model=None)
            self.backend_name = "sensevoice"
            print(f"[full_duplex] SenseVoice ASR ready: {model_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize SenseVoice from `{model_path}`. "
                "Realtime ASR will not fall back to Whisper anymore."
            ) from exc

    def transcribe(self, audio: np.ndarray, is_final: bool = True) -> str:
        self.load()
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return ""

        if self._funasr_model is not None:
            try:
                result = self._funasr_model.generate(input=audio, cache={}, is_final=is_final, language="auto")
                if isinstance(result, list) and result:
                    return clean_sensevoice_text(str(result[0].get("text", "")).strip())
                if isinstance(result, dict):
                    return clean_sensevoice_text(str(result.get("text", "")).strip())
            except Exception as exc:
                raise RuntimeError("SenseVoice transcription failed") from exc

        return ""


def clean_sensevoice_text(text: str) -> str:
    """Remove SenseVoice inline control tokens from transcripts."""
    return re.sub(r"<\|[^|]+?\|>", "", text).strip()


class QwenRealtimeBackend:
    """LLM adapter that reuses the mature conversation engine while enabling streaming."""

    def __init__(self) -> None:
        self.engine = ConversationEngine()
        self.model_name = os.environ.get("REALTIME_QWEN_MODEL_NAME", DEFAULT_REALTIME_QWEN_MODEL)
        self.local_model_path = os.environ.get("REALTIME_QWEN_MODEL_PATH") or _default_model_path("Qwen2.5-1.5B-Instruct")
        if not os.path.exists(self.local_model_path):
            self.local_model_path = _default_model_path("Qwen2.5-7B-Instruct")
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        local_path = self.local_model_path if os.path.exists(self.local_model_path) else None
        model_name = self.model_name if not local_path else None
        self.loaded = bool(self.engine.init_model(model_name=model_name, local_model_path=local_path))
        if not self.loaded:
            print(f"[full_duplex] Realtime Qwen init failed, legacy fallback: {DEFAULT_QWEN_MODEL}")

    def build_context_prompt(self, history: List[dict], user_text: str) -> str:
        prompt_parts: List[str] = []
        for message in history[-10:]:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            if content:
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    async def generate(self, history: List[dict], user_text: str) -> AsyncIterator[str]:
        self.load()
        context_prompt = self.build_context_prompt(history, user_text)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def worker() -> None:
            try:
                for chunk in self.engine.generate_response_stream(context_prompt, use_context=False):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.create_task(asyncio.to_thread(worker))

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


class CosyVoiceTTSBackend:
    """TTS adapter that prefers CosyVoice and falls back to the mature TTSEngine."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.model_name = os.environ.get("COSYVOICE_MODEL_NAME", DEFAULT_COSYVOICE_MODEL)
        self.local_model_path = os.environ.get("COSYVOICE_MODEL_PATH") or _default_model_path("CosyVoice-300M-SFT")
        self._cosyvoice = None
        self._fallback_tts: Optional[TTSEngine] = None
        self.backend_name = "cosyvoice"
        self.speaker = os.environ.get("COSYVOICE_SPEAKER")
        self.use_streaming = os.environ.get("COSYVOICE_STREAM", "1") != "0"

    def load(self) -> None:
        if self._cosyvoice is not None or self._fallback_tts is not None:
            return

        try:
            project_root = _project_root()
            local_cosyvoice_repo = os.environ.get("COSYVOICE_REPO_PATH") or os.path.join(project_root, "CosyVoice")
            if os.path.exists(local_cosyvoice_repo) and local_cosyvoice_repo not in sys.path:
                sys.path.insert(0, local_cosyvoice_repo)
            local_matcha = os.path.join(local_cosyvoice_repo, "third_party", "Matcha-TTS")
            if os.path.exists(local_matcha) and local_matcha not in sys.path:
                sys.path.insert(0, local_matcha)
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel
            except ImportError:
                from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceModel

            model_path = self.local_model_path if os.path.exists(self.local_model_path) else self.model_name
            use_fp16 = _should_use_fp16()
            self._cosyvoice = CosyVoiceModel(model_path, fp16=use_fp16)
            self.speaker = self._resolve_speaker()
            self.backend_name = "cosyvoice"
            print(f"[full_duplex] CosyVoice ready: {model_path}")
            print(f"[full_duplex] CosyVoice speaker: {self.speaker}")
            print(f"[full_duplex] CosyVoice fp16: {use_fp16}")
            return
        except Exception as exc:
            print(f"[full_duplex] CosyVoice unavailable, fallback to TTSEngine: {exc}")

        self._fallback_tts = TTSEngine(prefer_edge_tts=True, prefer_local_tts=False)
        self._fallback_tts.init_tts()
        self.backend_name = "legacy-tts"

    async def synthesize(self, text: str) -> SynthesizedSegment:
        self.load()
        if self._cosyvoice is not None:
            try:
                return await asyncio.to_thread(self._synthesize_with_cosyvoice, text)
            except Exception as exc:
                print(f"[full_duplex] CosyVoice synthesis failed, use fallback: {exc}")

        return await asyncio.to_thread(self._synthesize_with_legacy_tts, text)

    async def synthesize_stream(self, text: str) -> AsyncIterator[SynthesizedSegment]:
        """Yield TTS audio segments as soon as CosyVoice produces them."""
        self.load()
        if self._cosyvoice is None:
            yield await asyncio.to_thread(self._synthesize_with_legacy_tts, text)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[SynthesizedSegment]] = asyncio.Queue()

        def worker() -> None:
            start_time = time.time()
            first_segment = True
            try:
                for segment in self._synthesize_with_cosyvoice_stream(text):
                    if first_segment:
                        print(f"[full_duplex][TTS] first audio ready in {time.time() - start_time:.2f}s")
                        first_segment = False
                    asyncio.run_coroutine_threadsafe(queue.put(segment), loop)
            except Exception as exc:
                print(f"[full_duplex] CosyVoice streaming synthesis failed: {exc}")
                try:
                    fallback = self._synthesize_with_legacy_tts(text)
                    asyncio.run_coroutine_threadsafe(queue.put(fallback), loop)
                except Exception as fallback_exc:
                    print(f"[full_duplex] TTS fallback failed: {fallback_exc}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.create_task(asyncio.to_thread(worker))

        while True:
            segment = await queue.get()
            if segment is None:
                break
            yield segment

    def _synthesize_with_cosyvoice(self, text: str) -> SynthesizedSegment:
        generator = self._cosyvoice.inference_sft(text, self.speaker, stream=self.use_streaming)
        for item in generator:
            segment = self._segment_from_cosyvoice_item(text, item)
            if segment:
                return segment
        raise RuntimeError("CosyVoice returned no audio")

    def _synthesize_with_cosyvoice_stream(self, text: str) -> Iterable[SynthesizedSegment]:
        generator = self._cosyvoice.inference_sft(text, self.speaker, stream=self.use_streaming)
        emitted = False
        for item in generator:
            segment = self._segment_from_cosyvoice_item(text, item)
            if segment:
                emitted = True
                yield segment
        if not emitted:
            raise RuntimeError("CosyVoice returned no audio")

    def _segment_from_cosyvoice_item(self, text: str, item: dict) -> Optional[SynthesizedSegment]:
        waveform = item.get("tts_speech")
        if waveform is None:
            return None
        audio = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        audio = np.squeeze(audio).astype(np.float32)
        source_sample_rate = int(getattr(self._cosyvoice, "sample_rate", self.sample_rate))
        audio = _resample_audio(audio, source_sample_rate, self.sample_rate)
        return SynthesizedSegment(text=text, pcm_bytes=_float32_to_pcm16(audio), sample_rate=self.sample_rate)

    def _resolve_speaker(self) -> str:
        requested = os.environ.get("COSYVOICE_SPEAKER")
        available = []
        if hasattr(self._cosyvoice, "list_available_spks"):
            try:
                available = list(self._cosyvoice.list_available_spks())
            except Exception:
                available = []

        if requested:
            if not available or requested in available:
                return requested
            print(f"[full_duplex] Requested CosyVoice speaker `{requested}` not found. Available speakers: {available}")

        if available:
            return available[0]

        raise RuntimeError(
            "CosyVoice model has no SFT speakers. Use a model with `spk2info.pt` "
            "or configure zero-shot/cross-lingual synthesis."
        )

    def _synthesize_with_legacy_tts(self, text: str) -> SynthesizedSegment:
        if self._fallback_tts is None:
            self._fallback_tts = TTSEngine(prefer_edge_tts=True, prefer_local_tts=False)
            self._fallback_tts.init_tts()

        audio_file = self._fallback_tts.generate_speech_file(text, save_dir=tempfile.gettempdir())
        if not audio_file or not os.path.exists(audio_file):
            raise RuntimeError("Fallback TTSEngine did not create an audio file")

        try:
            pcm_bytes, sample_rate = _load_audio_file_as_pcm(audio_file, self.sample_rate)
            return SynthesizedSegment(text=text, pcm_bytes=pcm_bytes, sample_rate=sample_rate)
        finally:
            try:
                os.unlink(audio_file)
            except OSError:
                pass


class RealtimeBackendBundle:
    """Shared bundle of realtime-capable model adapters."""

    def __init__(self) -> None:
        self.asr = SenseVoiceASRBackend()
        self.llm = QwenRealtimeBackend()
        self.tts = CosyVoiceTTSBackend()

    def warmup(self) -> None:
        self.asr.load()
        self.llm.load()
        self.tts.load()


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)
    return int16_audio.tobytes()


def _should_use_fp16() -> bool:
    if os.environ.get("COSYVOICE_FP16", "1") == "0":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resample_audio(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32)
    from scipy import signal

    target_length = int(len(audio) * target_sample_rate / source_sample_rate)
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    return signal.resample(audio, target_length).astype(np.float32)


def _load_audio_file_as_pcm(file_path: str, target_sample_rate: int) -> tuple[bytes, int]:
    try:
        import soundfile as sf

        audio_data, sample_rate = sf.read(file_path, dtype="float32")
    except Exception:
        with wave.open(file_path, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sample_rate:
        from scipy import signal

        target_length = int(len(audio_data) * target_sample_rate / sample_rate)
        audio_data = signal.resample(audio_data, target_length).astype(np.float32)
        sample_rate = target_sample_rate

    return _float32_to_pcm16(np.asarray(audio_data, dtype=np.float32)), sample_rate
