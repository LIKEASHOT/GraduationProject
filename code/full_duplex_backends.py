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
import tempfile
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

import numpy as np
from config import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_REALTIME_QWEN_MODEL,
    DEFAULT_SENSEVOICE_MODEL,
    REALTIME_AUDIO_CHUNK_SAMPLES,
    REALTIME_TTS_AUDIO_CHUNK_SAMPLES,
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

    normalized = _normalize_tts_text_boundaries(normalized)
    protected = _extract_quoted_spans(normalized)
    if protected:
        segments = _split_mixed_text_with_quotes(normalized)
        return _merge_short_tts_prefixes(segments)

    sentence_breaks = "\u3002\uff01\uff1f!?"
    soft_breaks = "\uff0c,\uff1b;:\uff1a"
    pieces = re.split(rf"(?<=[{sentence_breaks}{soft_breaks}])\s+|(?<=[{sentence_breaks}{soft_breaks}])", normalized)
    segments: List[str] = []
    current = ""

    for piece in pieces:
        piece = _clean_tts_segment(piece)
        if not piece or _is_punctuation_only(piece):
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
        short_segments.extend(_split_long_tts_segment(segment, max_chars=24, max_words=12))
    return _merge_short_tts_prefixes(short_segments)


def _normalize_tts_text_boundaries(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'"\s*([^"]+?)\s*"', r'"\1"', text)
    return text


def _extract_quoted_spans(text: str) -> List[tuple[int, int, str]]:
    spans: List[tuple[int, int, str]] = []
    for match in re.finditer(r'"[^"]+"', text):
        quoted = match.group(0)
        inner = quoted.strip('"')
        if _should_protect_quoted_span(inner):
            spans.append((match.start(), match.end(), quoted))
    return spans


def _should_protect_quoted_span(text: str) -> bool:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if latin_count == 0:
        return False
    if cjk_count == 0:
        return True
    return latin_count >= cjk_count


def _split_mixed_text_with_quotes(text: str) -> List[str]:
    spans = _extract_quoted_spans(text)
    if not spans:
        return split_sentences(text.replace('"', ""))
    raw_segments: List[str] = []
    cursor = 0
    for start, end, quoted in spans:
        raw_segments.extend(_split_non_quoted_text(text[cursor:start]))
        raw_segments.append(_clean_tts_segment(quoted))
        cursor = end
    raw_segments.extend(_split_non_quoted_text(text[cursor:]))

    segments: List[str] = []
    for segment in raw_segments:
        segment = _clean_tts_segment(segment)
        if not segment or _is_punctuation_only(segment):
            continue
        if segment.startswith('"') and segment.endswith('"'):
            segments.append(segment)
            continue
        segments.extend(_split_long_tts_segment(segment, max_chars=24, max_words=12))
    return segments


def _split_non_quoted_text(text: str) -> List[str]:
    text = _clean_tts_segment(text)
    if not text:
        return []
    sentence_breaks = "\u3002\uff01\uff1f!?"
    soft_breaks = "\uff0c,\uff1b;:\uff1a"
    pieces = re.split(rf"(?<=[{sentence_breaks}{soft_breaks}])\s*|(?<=[{soft_breaks}])\s*", text)
    return [_clean_tts_segment(piece) for piece in pieces if _clean_tts_segment(piece)]


def _merge_short_tts_prefixes(segments: List[str]) -> List[str]:
    merged: List[str] = []
    pending = ""
    prefix_endings = tuple("\uff1a:，,")

    for segment in segments:
        segment = _clean_tts_segment(segment)
        if not segment or _is_punctuation_only(segment):
            continue

        if pending:
            segment = f"{pending} {segment}".strip()
            pending = ""

        plain_len = len(re.sub(r"\s+", "", segment))
        word_count = len(re.findall(r"[A-Za-z0-9]+", segment))
        if segment.endswith(prefix_endings) and (plain_len <= 12 or 0 < word_count <= 3):
            pending = segment
            continue

        merged.append(segment)

    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending.rstrip("\uff1a:，,"))
    return merged


def _clean_tts_segment(text: str) -> str:
    cleaned = str(text or "").strip().strip("\n\r\t ")
    return cleaned.strip('"“”')


def _is_punctuation_only(text: str) -> bool:
    return not re.search(r"[\w\u4e00-\u9fff]", text or "")


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _split_long_tts_segment(text: str, max_chars: int, max_words: int) -> List[str]:
    text = _clean_tts_segment(text)
    if not text or _is_punctuation_only(text):
        return []
    if text.startswith('"') and text.endswith('"'):
        return [text]
    if not _contains_cjk(text):
        return _split_english_tts_segment(text, max_words=max_words, max_chars=max_chars * 4)
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


def _split_english_tts_segment(text: str, max_words: int, max_chars: int) -> List[str]:
    tokens = re.findall(r'"[^"]+"|\S+', text)
    if not tokens:
        return []

    chunks: List[str] = []
    current_tokens: List[str] = []
    current_words = 0

    for token in tokens:
        token_words = len(re.findall(r"[A-Za-z0-9]+", token)) or 1
        candidate = " ".join(current_tokens + [token]).strip()
        should_flush = (
            current_tokens
            and (current_words + token_words > max_words or len(candidate) > max_chars)
        )
        if should_flush:
            chunks.append(" ".join(current_tokens).strip())
            current_tokens = [token]
            current_words = token_words
        else:
            current_tokens.append(token)
            current_words += token_words

    if current_tokens:
        chunks.append(" ".join(current_tokens).strip())

    return [chunk for chunk in chunks if chunk and not _is_punctuation_only(chunk)]


def split_sentences(text: str) -> List[str]:
    """Split TTS text by punctuation and budget without treating quotes as atomic."""
    normalized = TextProcessor.clean_response(text)
    if not normalized:
        return []

    normalized = _tts_normalize_text(normalized)
    primary_segments = _tts_split_by_punctuation(normalized)
    budgeted_segments: List[str] = []
    for segment in primary_segments or [normalized]:
        budgeted_segments.extend(_tts_split_by_budget(segment))
    return _tts_enforce_final_budget(_tts_merge_short_prefixes(budgeted_segments))


def _tts_normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text.strip('"')


def _tts_clean_segment(text: str) -> str:
    return str(text or "").strip().strip("\n\r\t ").strip('"')


def _tts_split_by_punctuation(text: str) -> List[str]:
    cjk_breaks = set("\u3002\uff01\uff1f\uff0c\uff1b\uff1a")
    sentence_breaks = set(".!?")
    soft_breaks = set(",;:")
    closers = set('"\'\u201d\u2019\uff09)]}\u300b\u300d\u300f')
    segments: List[str] = []
    current = ""

    for index, char in enumerate(text):
        current += char
        if char not in cjk_breaks and char not in sentence_breaks and char not in soft_breaks:
            continue
        if char == "." and _tts_is_decimal_or_abbreviation(text, index):
            continue

        if char in cjk_breaks:
            segment = _tts_clean_segment(current)
            if segment and not _is_punctuation_only(segment):
                segments.append(segment)
            current = ""
            continue

        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char in soft_breaks and len(current.strip()) < 8:
            continue
        if next_char and next_char not in " \t\r\n" and next_char not in closers:
            continue

        segment = _tts_clean_segment(current)
        if segment and not _is_punctuation_only(segment):
            segments.append(segment)
        current = ""

    tail = _tts_clean_segment(current)
    if tail and not _is_punctuation_only(tail):
        segments.append(tail)
    return segments


def _tts_is_decimal_or_abbreviation(text: str, index: int) -> bool:
    previous = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and next_char.isdigit():
        return True
    if previous.isupper() and (not next_char or next_char.isspace()):
        return True
    return False


def _tts_split_by_budget(text: str) -> List[str]:
    text = _tts_clean_segment(text)
    if not text or _is_punctuation_only(text):
        return []

    max_cjk_chars = int(os.environ.get("REALTIME_TTS_MAX_CJK_CHARS", "34"))
    max_en_words = int(os.environ.get("REALTIME_TTS_MAX_EN_WORDS", "14"))
    max_mixed_chars = int(os.environ.get("REALTIME_TTS_MAX_MIXED_CHARS", "56"))

    latin_words = _tts_latin_word_count(text)
    has_cjk = _contains_cjk(text)
    if not has_cjk:
        return _tts_split_latin_by_words(text, max_words=max_en_words, max_chars=max_mixed_chars * 2)
    if len(text) <= max_cjk_chars and latin_words <= max_en_words:
        return [text]
    if latin_words >= 4:
        return _tts_split_mixed_by_tokens(text, max_chars=max_mixed_chars, max_words=max_en_words)
    return _tts_split_cjk_by_chars(text, max_chars=max_cjk_chars)


def _tts_split_cjk_by_chars(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) < max_chars:
            continue
        chunk = _tts_clean_segment(current)
        if chunk and not _is_punctuation_only(chunk):
            chunks.append(chunk)
        current = ""
    tail = _tts_clean_segment(current)
    if tail and not _is_punctuation_only(tail):
        chunks.append(tail)
    return chunks


def _tts_merge_tiny_chunks(chunks: List[str]) -> List[str]:
    if len(chunks) <= 1:
        return chunks

    merged: List[str] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        word_count = _tts_latin_word_count(chunk)
        if word_count and word_count <= 3:
            if index + 1 < len(chunks):
                merged.append(_tts_join_segments(chunk, chunks[index + 1]))
                index += 2
                continue
            if merged:
                merged[-1] = _tts_join_segments(merged[-1], chunk)
                index += 1
                continue
        merged.append(chunk)
        index += 1
    return merged


def _tts_split_mixed_by_tokens(text: str, max_chars: int, max_words: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]+|[^\s]", text)
    return _tts_pack_tokens(tokens, max_chars=max_chars, max_words=max_words)


def _tts_split_latin_by_words(text: str, max_words: int, max_chars: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", text)
    return _tts_pack_tokens(tokens, max_chars=max_chars, max_words=max_words)


def _tts_pack_tokens(tokens: List[str], max_chars: int, max_words: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    current_words = 0

    for token in tokens:
        token_words = 1 if re.match(r"[A-Za-z0-9]", token) else 0
        separator = " " if current and _tts_needs_space(current[-1], token[0]) else ""
        candidate = f"{current}{separator}{token}"
        should_flush = current and (len(candidate) > max_chars or current_words + token_words > max_words)
        if should_flush:
            chunk = _tts_clean_segment(current)
            if chunk and not _is_punctuation_only(chunk):
                chunks.append(chunk)
            current = token
            current_words = token_words
        else:
            current = candidate
            current_words += token_words

    tail = _tts_clean_segment(current)
    if tail and not _is_punctuation_only(tail):
        chunks.append(tail)
    return _tts_merge_tiny_chunks(chunks)


def _tts_needs_space(left: str, right: str) -> bool:
    if right in ".,!?;:%)]}":
        return False
    if left in "([{$\"'":
        return False
    if right in "\"'":
        return False
    return bool(re.match(r"[A-Za-z0-9]", left) or re.match(r"[A-Za-z0-9]", right))


def _tts_latin_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))


def _tts_is_list_marker(text: str) -> bool:
    return bool(re.fullmatch(r"\d+[\.)\u3001\uff0e]?", _tts_clean_segment(text)))


def _tts_merge_short_prefixes(segments: List[str]) -> List[str]:
    merged: List[str] = []
    pending = ""
    prefix_endings = ("\uff1a", ":", "\uff0c", ",", "\uff1b", ";")

    for segment in segments:
        segment = _tts_clean_segment(segment)
        if not segment or _is_punctuation_only(segment):
            continue

        if _tts_is_list_marker(segment):
            pending = _tts_join_segments(pending, segment) if pending else segment
            continue

        if pending:
            segment = _tts_join_segments(pending, segment)
            pending = ""

        plain_len = len(re.sub(r"\s+", "", segment))
        word_count = _tts_latin_word_count(segment)
        if segment.endswith(prefix_endings) and (plain_len <= 16 or 0 < word_count <= 3):
            pending = segment
            continue

        merged.append(segment)

    if pending:
        if merged:
            merged[-1] = _tts_join_segments(merged[-1], pending)
        else:
            merged.append(pending.rstrip("\uff1a:\uff0c,\uff1b;"))
    return merged


def _tts_join_segments(left: str, right: str) -> str:
    left = _tts_clean_segment(left)
    right = _tts_clean_segment(right)
    if not left:
        return right
    if not right:
        return left
    if left[-1] in "\u3002\uff01\uff1f\uff0c\uff1b\uff1a":
        return f"{left}{right}"
    return f"{left} {right}".strip()


def _tts_enforce_final_budget(segments: List[str]) -> List[str]:
    final_segments: List[str] = []
    for segment in segments:
        final_segments.extend(_tts_split_by_budget(segment))
    return [segment for segment in final_segments if segment and not _is_punctuation_only(segment)]


@dataclass
class SynthesizedSegment:
    text: str
    pcm_bytes: bytes
    sample_rate: int
    audio_format: str = "pcm"

    def iter_base64_chunks(self, chunk_samples: int = REALTIME_TTS_AUDIO_CHUNK_SAMPLES) -> Iterable[str]:
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

    async def generate_full(self, history: List[dict], user_text: str) -> str:
        """Generate a complete response before TTS starts."""
        self.load()
        context_prompt = self.build_context_prompt(history, user_text)
        return await asyncio.to_thread(
            self.engine.generate_response,
            context_prompt,
            512,
            False,
            True,
            False,
        )


class RealtimeTTSBackend:
    """Realtime TTS adapter using the previous stable Edge/gTTS stack only."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._tts: Optional[TTSEngine] = None
        self.backend_name = "edge/gtts"

    def load(self) -> None:
        if self._tts is not None:
            return

        self._tts = TTSEngine(prefer_edge_tts=True, prefer_local_tts=True)
        self._tts.init_tts()
        print(f"[full_duplex] Realtime TTS engine: {self._tts.get_current_engine_info()}")

    async def synthesize(self, text: str) -> SynthesizedSegment:
        self.load()
        return await asyncio.to_thread(self._synthesize_with_stable_tts, text)

    async def synthesize_stream(self, text: str) -> AsyncIterator[SynthesizedSegment]:
        """Keep the realtime API shape while using the stable file-based TTS path."""
        self.load()
        yield await asyncio.to_thread(self._synthesize_with_stable_tts, text)

    def _synthesize_with_stable_tts(self, text: str) -> SynthesizedSegment:
        if self._tts is None:
            self.load()
        if self._tts is None:
            raise RuntimeError("Stable TTSEngine is not initialized")

        audio_file = self._tts.generate_speech_file(text, save_dir=tempfile.gettempdir())
        if not audio_file or not os.path.exists(audio_file):
            raise RuntimeError("TTSEngine did not create an audio file")

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
        self.tts = RealtimeTTSBackend()

    def warmup(self) -> None:
        self.asr.load()
        self.llm.load()
        self.tts.load()


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)
    return int16_audio.tobytes()


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

    audio_data = _trim_tts_silence(audio_data, sample_rate)

    return _float32_to_pcm16(np.asarray(audio_data, dtype=np.float32)), sample_rate


def _trim_tts_silence(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading/trailing silence that makes segmented TTS sound choppy."""
    if os.environ.get("DISABLE_TTS_SILENCE_TRIM", "0") == "1":
        return audio_data

    audio = np.asarray(audio_data, dtype=np.float32).flatten()
    if audio.size == 0:
        return audio

    threshold = float(os.environ.get("TTS_SILENCE_TRIM_THRESHOLD", "0.008"))
    keep_ms = int(os.environ.get("TTS_SILENCE_KEEP_MS", "40"))
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio

    indices = np.flatnonzero(mask)
    keep_samples = max(0, sample_rate * keep_ms // 1000)
    start = max(0, int(indices[0]) - keep_samples)
    end = min(audio.size, int(indices[-1]) + keep_samples)
    trimmed = audio[start:end]
    if trimmed.size < sample_rate * 0.12:
        return audio
    return trimmed
