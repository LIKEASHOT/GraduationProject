#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime VAD helpers for pseudo full-duplex sessions.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from config import (
    REALTIME_INTERRUPT_GRACE_MS,
    REALTIME_INTERRUPT_THRESHOLD,
    REALTIME_VAD_END_THRESHOLD,
    REALTIME_VAD_START_THRESHOLD,
    SAMPLE_RATE,
)


@dataclass
class VADDecision:
    speech_prob: float
    energy: float
    is_speech: bool
    speech_started: bool
    speech_ended: bool
    should_interrupt: bool
    echo_similarity: float


class RealtimeVAD:
    """Wrap Silero VAD when available and fall back to an energy-based heuristic."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        start_threshold: float = REALTIME_VAD_START_THRESHOLD,
        end_threshold: float = REALTIME_VAD_END_THRESHOLD,
        interrupt_threshold: float = REALTIME_INTERRUPT_THRESHOLD,
        interrupt_grace_ms: int = REALTIME_INTERRUPT_GRACE_MS,
        silero_repo: Optional[str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.interrupt_threshold = interrupt_threshold
        self.interrupt_grace_ms = interrupt_grace_ms
        self.silero_repo = silero_repo

        self._model = None
        self._torch = None
        self._in_speech = False
        self._last_tts_frame_at = -10**9
        self._frame_index = 0
        self._recent_probs: Deque[float] = deque(maxlen=6)
        self._recent_echo_scores: Deque[float] = deque(maxlen=4)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch

            source = "github"
            repo_or_dir = self.silero_repo or "snakers4/silero-vad"
            if repo_or_dir and ("/" not in repo_or_dir and "\\" in repo_or_dir):
                source = "local"
            self._model, _ = torch.hub.load(
                repo_or_dir=repo_or_dir,
                model="silero_vad",
                source=source,
                trust_repo=True,
            )
            self._torch = torch
        except Exception as exc:
            print(f"[full_duplex] Silero VAD unavailable, fallback to RMS heuristic: {exc}")
            self._model = None
            self._torch = None

    def mark_tts_frame(self) -> None:
        self._last_tts_frame_at = self._frame_index

    def analyze(
        self,
        audio: np.ndarray,
        reference_tts: Optional[np.ndarray] = None,
        is_ai_speaking: bool = False,
    ) -> VADDecision:
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return VADDecision(0.0, 0.0, False, False, False, False, 0.0)

        self._frame_index += 1
        speech_prob = self._speech_probability(audio)
        energy = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        self._recent_probs.append(speech_prob)
        smoothed_prob = float(sum(self._recent_probs) / len(self._recent_probs))

        speech_started = False
        speech_ended = False
        if not self._in_speech and smoothed_prob >= self.start_threshold:
            self._in_speech = True
            speech_started = True
        elif self._in_speech and smoothed_prob <= self.end_threshold:
            self._in_speech = False
            speech_ended = True

        echo_similarity = self._echo_similarity(audio, reference_tts)
        self._recent_echo_scores.append(echo_similarity)
        echo_penalty = float(sum(self._recent_echo_scores) / len(self._recent_echo_scores))

        grace_frames = max(
            1,
            int(math.ceil((self.interrupt_grace_ms / 1000.0) * self.sample_rate / max(audio.size, 1))),
        )
        in_grace_window = (self._frame_index - self._last_tts_frame_at) <= grace_frames
        interrupt_score = smoothed_prob + min(0.35, energy * 8.0) - (echo_penalty * 0.45)
        should_interrupt = bool(
            is_ai_speaking and not in_grace_window and interrupt_score >= self.interrupt_threshold
        )

        return VADDecision(
            speech_prob=smoothed_prob,
            energy=energy,
            is_speech=self._in_speech,
            speech_started=speech_started,
            speech_ended=speech_ended,
            should_interrupt=should_interrupt,
            echo_similarity=echo_similarity,
        )

    def _speech_probability(self, audio: np.ndarray) -> float:
        self.load()
        if self._model is not None and self._torch is not None:
            try:
                tensor = self._torch.from_numpy(audio)
                return float(self._model(tensor, self.sample_rate).item())
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
        return min(1.0, max(0.0, rms * 18.0))

    @staticmethod
    def _echo_similarity(audio: np.ndarray, reference_tts: Optional[np.ndarray]) -> float:
        if reference_tts is None or reference_tts.size == 0:
            return 0.0

        target = reference_tts[-audio.size:]
        if target.size != audio.size:
            return 0.0

        audio_norm = np.linalg.norm(audio)
        target_norm = np.linalg.norm(target)
        if audio_norm == 0.0 or target_norm == 0.0:
            return 0.0

        similarity = float(np.dot(audio, target) / (audio_norm * target_norm))
        return max(0.0, min(1.0, similarity))
