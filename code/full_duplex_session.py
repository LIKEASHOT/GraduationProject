#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session orchestration for the pseudo full-duplex websocket backend.
Frontend-owned VAD means the server receives a whole utterance and a
separate `speech_end` event before running ASR -> LLM -> TTS.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from starlette.websockets import WebSocketDisconnect

from config import REALTIME_MAX_HISTORY_TURNS, REALTIME_MIN_UTTERANCE_MS, SAMPLE_RATE
from dialogue_policy import DialoguePolicy
from full_duplex_backends import RealtimeBackendBundle, SynthesizedSegment, split_sentences


def _normalize_model_variant(payload: dict, default_variant: str = "base") -> str:
    raw_variant = (
        payload.get("model_variant")
        or payload.get("model_mode")
        or payload.get("llm_mode")
        or payload.get("model")
        or default_variant
    )
    if payload.get("use_lora") is True:
        raw_variant = "lora"
    variant = str(raw_variant or "base").strip().lower()
    if variant in {"lora", "finetuned", "fine_tuned", "ft", "sft"}:
        return "lora"
    return "base"


class FullDuplexSession:
    """One websocket session with frontend-driven VAD boundaries."""

    def __init__(self, websocket, backends: RealtimeBackendBundle) -> None:
        self.websocket = websocket
        self.backends = backends
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.turn_index = 0
        self.history: List[Dict[str, str]] = []

        self._current_audio_frames: List[np.ndarray] = []
        self._current_partial_text = ""
        self._muted = False
        self._ended = False
        self._state = "IDLE"

        self._active_turn_id: Optional[str] = None
        self._active_message_id: Optional[str] = None
        self._assistant_dispatched_segments: List[str] = []
        self._assistant_sentence_buffer = ""

        self._llm_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._turn_lock = asyncio.Lock()
        self._turn_end_sent = False
        self._closed = False
        self._model_variant = getattr(self.backends.llm, "default_model_variant", "base")
        self._stream_text_delta = os.environ.get("REALTIME_LLM_TEXT_STREAM", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    async def run(self) -> None:
        await self._send(
            "session_ready",
            {
                "session_id": self.session_id,
                "capabilities": {
                    "text_chat_preserved": True,
                    "speech_to_text_preserved": True,
                    "tts_preserved": True,
                    "full_duplex": True,
                    "vad_owner": "frontend",
                    "llm_text_stream": self._stream_text_delta,
                    "model_variants": ["base", "lora"],
                },
                "model_variant": self._model_variant,
            },
        )

        try:
            while not self._ended:
                raw_message = await self.websocket.receive_text()
                await self._handle_client_event(raw_message)
        except WebSocketDisconnect:
            self._closed = True
        finally:
            await self._shutdown()

    async def _handle_client_event(self, raw_message: str) -> None:
        message = json.loads(raw_message)
        event = message.get("event")
        payload = message.get("payload", {})

        if event == "audio_stream":
            await self._on_audio_chunk(payload)
        elif event == "speech_end":
            await self._on_speech_end()
        elif event == "interrupt":
            await self._interrupt_current_turn(reason=payload.get("reason", "client_interrupt"), notify_client=False)
        elif event == "mute_mic":
            self._muted = bool(payload.get("status", False))
        elif event == "end_call":
            self._ended = True
        elif event == "session_start":
            self._model_variant = _normalize_model_variant(payload, self._model_variant)
            await self._send(
                "session_ready",
                {
                    "session_id": self.session_id,
                    "model_variant": self._model_variant,
                },
            )
        else:
            await self._send("error", {"code": "UNKNOWN_EVENT", "message": f"Unsupported event: {event}"})

    async def _on_audio_chunk(self, payload: dict) -> None:
        if self._muted:
            return

        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))
        pcm = _decode_audio_chunk(payload.get("audio_chunk", ""), payload.get("format", "pcm"))
        if pcm.size == 0:
            return

        if sample_rate != SAMPLE_RATE:
            from scipy import signal

            target_length = int(len(pcm) * SAMPLE_RATE / sample_rate)
            pcm = signal.resample(pcm, target_length).astype(np.float32)

        self._current_audio_frames.append(pcm)
        self._state = "USER_BUFFERING"

    async def _on_speech_end(self) -> None:
        if self._state == "AI_SPEAKING":
            await self._interrupt_current_turn(reason="user_speaking", notify_client=False)

        audio = np.concatenate(self._current_audio_frames) if self._current_audio_frames else np.array([], dtype=np.float32)
        self._current_audio_frames = []
        self._current_partial_text = ""

        min_samples = int(SAMPLE_RATE * REALTIME_MIN_UTTERANCE_MS / 1000.0)
        if audio.size < min_samples:
            self._state = "IDLE"
            return

        self._state = "THINKING"
        transcript = await asyncio.to_thread(self.backends.asr.transcribe, audio, True)
        transcript = transcript.strip()
        if not transcript:
            self._state = "IDLE"
            return

        print(f"[full_duplex][ASR] user_text: {transcript}")
        self.history.append({"role": "user", "content": transcript})
        self.history = self.history[-REALTIME_MAX_HISTORY_TURNS * 2 :]
        await self._prepare_assistant_turn()
        await self._send(
            "user_text",
            {
                "turn_id": self._active_turn_id,
                "message_id": f"{self._active_message_id}_user" if self._active_message_id else None,
                "text": transcript,
                "is_final": True,
            },
        )
        await self._start_assistant_generation(transcript)

    async def _prepare_assistant_turn(self) -> None:
        async with self._turn_lock:
            await self._cancel_generation_tasks()

            self.turn_index += 1
            self._active_turn_id = f"turn_{self.turn_index:04d}"
            self._active_message_id = f"msg_{self.turn_index:04d}"
            self._assistant_dispatched_segments = []
            self._assistant_sentence_buffer = ""
            self._turn_end_sent = False
            self._state = "AI_SPEAKING"
            self._tts_queue = asyncio.Queue()
            self._tts_task = asyncio.create_task(self._tts_sender_loop())

    async def _start_assistant_generation(self, user_text: str) -> None:
        async with self._turn_lock:
            self._llm_task = asyncio.create_task(self._generate_assistant_reply(user_text))

    async def _generate_assistant_reply(self, user_text: str) -> None:
        turn_id = self._active_turn_id
        message_id = self._active_message_id
        tts_queue = self._tts_queue
        try:
            response_parts: List[str] = []

            async for chunk in self.backends.llm.generate(self.history, user_text, model_variant=self._model_variant):
                if not self._is_active_turn(turn_id, message_id):
                    return

                delta = str(chunk or "")
                if not delta:
                    continue

                current_response = "".join(response_parts)
                response_parts.append(_join_text_delta(current_response, delta))
                print(f"[full_duplex][LLM] delta: {delta}", end="", flush=True)
                if self._stream_text_delta:
                    await self._send(
                        "ai_text_delta",
                        {
                            "turn_id": turn_id,
                            "message_id": message_id,
                            "delta": delta,
                            "is_final": False,
                        },
                    )

            response_text = "".join(response_parts).strip()
            if not self._is_active_turn(turn_id, message_id) or not response_text:
                if self._is_active_turn(turn_id, message_id):
                    await tts_queue.put(None)
                return

            print()
            print(f"[full_duplex][LLM] full response: {response_text}")
            if os.environ.get("DIALOGUE_POLICY_REPAIR_ENABLED", "1").lower() in {"1", "true", "yes", "on"}:
                validation_prompt = self.backends.llm.build_context_prompt(self.history, user_text)
                direct_response = DialoguePolicy.direct_response(validation_prompt)
                if direct_response:
                    response_text = direct_response
                if DialoguePolicy.response_needs_retry(validation_prompt, response_text):
                    print("[full_duplex][LLM] response rejected by dialogue policy, switching to validated full generation...")
                    response_text = await self.backends.llm.generate_full(
                        self.history,
                        user_text,
                        model_variant=self._model_variant,
                    )
                    if DialoguePolicy.response_needs_retry(validation_prompt, response_text):
                        print("[full_duplex][LLM] validated full generation still rejected, using dialogue fallback.")
                        response_text = DialoguePolicy.fallback_response(validation_prompt)
            await self._send(
                "ai_text",
                {
                    "turn_id": turn_id,
                    "message_id": message_id,
                    "text": response_text,
                    "is_final": True,
                    "replace": True,
                    "model_variant": self._model_variant,
                },
            )

            for segment in split_sentences(response_text):
                if not self._is_active_turn(turn_id, message_id):
                    break
                print(f"[full_duplex][TTS] queued segment: {segment}")
                await tts_queue.put(segment)

            await tts_queue.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._send("error", {"code": "LLM_FAILED", "message": str(exc)})
            await tts_queue.put(None)

    async def _tts_sender_loop(self) -> None:
        try:
            while True:
                segment_text = await self._tts_queue.get()
                if segment_text is None:
                    break
                print(f"[full_duplex][TTS] synthesizing: {segment_text}")
                async for synthesized in self.backends.tts.synthesize_stream(segment_text):
                    await self._stream_tts_segment(synthesized)
        except asyncio.CancelledError:
            raise
        finally:
            if self._active_turn_id and self._state != "INTERRUPTING" and not self._closed:
                await self._finalize_assistant_history("completed")
                self._state = "IDLE"

    async def _stream_tts_segment(self, synthesized: SynthesizedSegment) -> None:
        segment_text = synthesized.text.strip()
        if not segment_text:
            return

        if os.environ.get("REALTIME_TTS_SEND_MODE", "segment").lower() != "chunk":
            await self._send_tts_segment_as_single_chunk(synthesized, segment_text)
            return

        first_chunk = True
        chunk_index = 0
        print(
            f"[full_duplex][TTS] streaming audio: text='{segment_text}', "
            f"sample_rate={synthesized.sample_rate}, bytes={len(synthesized.pcm_bytes)}"
        )
        for chunk_b64 in synthesized.iter_base64_chunks():
            chunk_index += 1
            await self._send(
                "tts_audio_chunk",
                {
                    "turn_id": self._active_turn_id,
                    "message_id": self._active_message_id,
                    "audio": chunk_b64,
                    "format": synthesized.audio_format,
                    "sample_rate": synthesized.sample_rate,
                    "chunk_id": f"{self._active_turn_id}_{chunk_index}",
                    "is_last_chunk": False,
                    "text_span": segment_text,
                },
            )
            if first_chunk:
                self._assistant_dispatched_segments.append(segment_text)
                first_chunk = False
            await asyncio.sleep(0)
        print(f"[full_duplex][TTS] sent {chunk_index} audio chunks")

        await self._send(
            "tts_audio_chunk",
            {
                "turn_id": self._active_turn_id,
                "message_id": self._active_message_id,
                "audio": "",
                "format": synthesized.audio_format,
                "sample_rate": synthesized.sample_rate,
                "chunk_id": f"{self._active_turn_id}_{chunk_index + 1}",
                "is_last_chunk": True,
                "text_span": segment_text,
            },
        )

    async def _send_tts_segment_as_single_chunk(self, synthesized: SynthesizedSegment, segment_text: str) -> None:
        duration = 0.0
        if synthesized.sample_rate > 0:
            duration = len(synthesized.pcm_bytes) / 2 / synthesized.sample_rate

        print(
            f"[full_duplex][TTS] sending segment audio: text='{segment_text}', "
            f"sample_rate={synthesized.sample_rate}, bytes={len(synthesized.pcm_bytes)}, "
            f"duration={duration:.2f}s"
        )
        await self._send(
            "tts_audio_chunk",
            {
                "turn_id": self._active_turn_id,
                "message_id": self._active_message_id,
                "audio": base64.b64encode(synthesized.pcm_bytes).decode("utf-8"),
                "format": synthesized.audio_format,
                "sample_rate": synthesized.sample_rate,
                "chunk_id": f"{self._active_turn_id}_segment_{len(self._assistant_dispatched_segments) + 1}",
                "is_last_chunk": True,
                "text_span": segment_text,
                "send_mode": "segment",
                "duration": round(duration, 3),
            },
        )
        self._assistant_dispatched_segments.append(segment_text)

    async def _interrupt_current_turn(self, reason: str, notify_client: bool) -> None:
        if self._state != "AI_SPEAKING":
            return

        self._state = "INTERRUPTING"
        await self._cancel_generation_tasks()
        if notify_client:
            await self._send(
                "interrupt",
                {
                    "turn_id": self._active_turn_id,
                    "reason": reason,
                    "stop_audio_playback": True,
                },
            )
        await self._finalize_assistant_history("interrupted")
        self._assistant_sentence_buffer = ""
        self._assistant_dispatched_segments = []
        self._active_turn_id = None
        self._active_message_id = None
        self._state = "IDLE"

    def _is_active_turn(self, turn_id: Optional[str], message_id: Optional[str]) -> bool:
        return (
            self._state == "AI_SPEAKING"
            and turn_id is not None
            and message_id is not None
            and self._active_turn_id == turn_id
            and self._active_message_id == message_id
            and not self._closed
        )

    async def _finalize_assistant_history(self, finish_reason: str) -> None:
        if self._turn_end_sent:
            return
        self._turn_end_sent = True

        dispatched_text = "".join(self._assistant_dispatched_segments).strip()
        if dispatched_text:
            self.history.append({"role": "assistant", "content": dispatched_text})
            self.history = self.history[-REALTIME_MAX_HISTORY_TURNS * 2 :]

        await self._send(
            "turn_end",
            {
                "turn_id": self._active_turn_id,
                "finish_reason": finish_reason,
            },
        )

    async def _cancel_generation_tasks(self) -> None:
        for task in (self._llm_task, self._tts_task):
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._llm_task = None
        self._tts_task = None

    async def _shutdown(self) -> None:
        self._closed = True
        await self._cancel_generation_tasks()
        if self.websocket.client_state.name != "DISCONNECTED":
            try:
                await self.websocket.close()
            except RuntimeError:
                pass

    async def _send(self, event: str, payload: dict) -> None:
        if self._closed:
            return
        try:
            await self.websocket.send_json({"event": event, "payload": payload})
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True


def _decode_audio_chunk(payload: str, audio_format: str) -> np.ndarray:
    if not payload:
        return np.array([], dtype=np.float32)

    raw = base64.b64decode(payload)
    if audio_format.lower() == "pcm":
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio_format.lower() == "wav":
        audio, _ = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32)
    return np.array([], dtype=np.float32)


def _extract_ready_segments(buffer: str) -> Tuple[List[str], str]:
    segments = split_sentences(buffer)
    if not segments:
        return [], ""
    if len(segments) == 1 and not _looks_complete(segments[0]):
        return [], buffer
    if len(segments) == 1:
        return segments, ""
    if _looks_complete(segments[-1]):
        return segments, ""
    return segments[:-1], segments[-1]


def _looks_complete(text: str) -> bool:
    punctuation = "\u3002\uff01\uff1f!?.,;:\uff0c\uff1b\uff1a"
    return bool(text.strip()) and text.strip()[-1] in punctuation


def _join_text_delta(previous_text: str, delta: str) -> str:
    delta = str(delta or "")
    if not previous_text or not delta:
        return delta

    previous_char = previous_text[-1]
    next_char = delta[0]
    if delta.startswith((" ", "\n", "\t")):
        return delta
    if next_char in ".,!?;:%)]}\"'":
        return delta
    if previous_char in " \n\t([{\"'":
        return delta
    if previous_char in ".!?" and _is_ascii_word_char(next_char):
        return f" {delta}"
    if _is_ascii_word_char(previous_char) and _is_ascii_word_char(next_char):
        return f" {delta}"
    return delta


def _is_ascii_word_char(char: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9]", char))
