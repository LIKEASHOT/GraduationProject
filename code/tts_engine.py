#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-to-speech engine.

Priority:
1. MOSS-TTS-Nano loaded directly in the backend process.
2. Microsoft Edge TTS as network fallback.
3. gTTS as last network fallback.

MOSS-TTS-Nano is loaded lazily and kept resident after the first initialization,
so realtime/full-duplex TTS does not pay process startup or model reload costs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

from language_utils import LanguageUtils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache")
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))
os.environ.setdefault("APPDATA", os.path.join(CACHE_ROOT, "appdata"))
os.makedirs(os.environ["APPDATA"], exist_ok=True)

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class TTSEngine:
    """TTS engine with MOSS-TTS-Nano first and network engines as fallbacks."""

    def __init__(self, prefer_edge_tts: bool = True, prefer_local_tts: bool = True):
        self.user_prefer_edge_tts = prefer_edge_tts
        self.user_prefer_local_tts = prefer_local_tts
        self.prefer_moss_tts = False
        self.prefer_edge_tts = False
        self.use_gtts = GTTS_AVAILABLE and os.environ.get("DISABLE_GTTS", "0") != "1"
        self.current_engine: Optional[str] = None

        self.moss_repo_path = os.environ.get(
            "MOSS_TTS_REPO_PATH",
            os.path.join(PROJECT_ROOT, "models", "MOSS-TTS-Nano"),
        )
        if not os.path.exists(self.moss_repo_path):
            self.moss_repo_path = os.path.join(PROJECT_ROOT, "MOSS-TTS-Nano")
        self.moss_model_path = os.environ.get(
            "MOSS_TTS_MODEL_PATH",
            os.environ.get(
                "MOSS_TTS_CHECKPOINT_PATH",
                os.path.join(PROJECT_ROOT, "models", "MOSS-TTS-Nano-Checkpoint"),
            ),
        )
        self.moss_audio_tokenizer_path = os.environ.get(
            "MOSS_TTS_AUDIO_TOKENIZER_PATH",
            os.path.join(PROJECT_ROOT, "models", "MOSS-Audio-Tokenizer-Nano"),
        )
        self.moss_model_name = os.environ.get("MOSS_TTS_MODEL_NAME", "OpenMOSS-Team/MOSS-TTS-Nano")
        self.moss_device = os.environ.get("MOSS_TTS_DEVICE", "")
        self.moss_dtype = os.environ.get("MOSS_TTS_DTYPE", "auto")
        self.moss_attn_implementation = os.environ.get("MOSS_TTS_ATTN_IMPLEMENTATION", "auto")
        self.moss_max_new_frames = int(os.environ.get("MOSS_TTS_MAX_NEW_FRAMES", "375"))
        self.moss_audio_temperature = float(os.environ.get("MOSS_TTS_AUDIO_TEMPERATURE", "0.8"))
        self.moss_speed = float(os.environ.get("MOSS_TTS_SPEED", "1.1"))
        self.moss_local_files_only = os.environ.get("MOSS_TTS_LOCAL_FILES_ONLY", "1") != "0"
        self.moss_eager_load = os.environ.get("MOSS_TTS_EAGER_LOAD", "1") != "0"
        self.moss_mode = os.environ.get("MOSS_TTS_MODE", "voice_clone")
        self.moss_voice = os.environ.get("MOSS_TTS_VOICE", "Lingyu")
        self.moss_en_voice = os.environ.get("MOSS_TTS_EN_VOICE", self.moss_voice)
        self.moss_zh_voice = os.environ.get("MOSS_TTS_ZH_VOICE", self.moss_voice)
        self.moss_voice_clone_max_text_tokens = int(os.environ.get("MOSS_TTS_VOICE_CLONE_MAX_TEXT_TOKENS", "75"))
        self._moss_model = None
        self._moss_processor = None
        self._moss_service = None
        self._moss_torch = None
        self._moss_torchaudio = None
        self._moss_runtime = "direct"

        self.moss_python = os.environ.get("MOSS_TTS_PYTHON") or shutil.which("python")
        self.moss_cli = os.environ.get("MOSS_TTS_CLI") or shutil.which("moss-tts-nano")
        self.moss_command = os.environ.get("MOSS_TTS_COMMAND")
        self.moss_timeout = float(os.environ.get("MOSS_TTS_TIMEOUT", "120"))
        self.moss_reference_audio = os.environ.get("MOSS_TTS_REFERENCE_AUDIO", "")

    def init_tts(self) -> bool:
        self.prefer_moss_tts = (
            self.user_prefer_local_tts
            and os.environ.get("DISABLE_MOSS_TTS", "0") != "1"
            and self._moss_configured()
        )
        self.prefer_edge_tts = bool(self.user_prefer_edge_tts and EDGE_TTS_AVAILABLE)

        if self.prefer_moss_tts and self.moss_eager_load:
            try:
                self._load_moss_direct()
                self._moss_runtime = "direct"
            except Exception as exc:
                if os.environ.get("ENABLE_MOSS_TTS_PROCESS_FALLBACK", "0") == "1" and self._moss_process_configured():
                    print(f"MOSS-TTS-Nano direct load failed, using process fallback: {exc!r}")
                    self._moss_runtime = "process"
                else:
                    print(f"MOSS-TTS-Nano direct load failed, disabling local TTS: {exc!r}")
                    self.prefer_moss_tts = False

        print(
            "TTS engine config - "
            f"MOSS-TTS-Nano first: {self.prefer_moss_tts}, "
            f"Edge fallback: {self.prefer_edge_tts}, "
            f"gTTS fallback: {self.use_gtts}"
        )

        if self.prefer_moss_tts:
            self.current_engine = "moss-tts-nano"
            print(f"Using MOSS-TTS-Nano {self._moss_runtime}: {self._moss_runtime_label()}")
            return True

        if self.prefer_edge_tts:
            self.current_engine = "edge"
            print("Using Edge TTS engine")
            return True

        if self.use_gtts:
            self.current_engine = "gtts"
            print("Using gTTS engine")
            return True

        print("No available TTS engine")
        return False

    def get_current_engine_info(self) -> str:
        engine_map = {
            "moss-tts-nano": "MOSS-TTS-Nano local",
            "edge": "Microsoft Edge TTS",
            "gtts": "Google gTTS",
        }
        return engine_map.get(self.current_engine, "Unknown TTS engine")

    def text_to_speech(self, text: str, async_play: bool = True, language: Optional[str] = None) -> None:
        file_path = self.generate_speech_file(text)
        if not file_path:
            return

        if async_play:
            thread = threading.Thread(target=self._play_audio_file, args=(file_path,), daemon=True)
            thread.start()
        else:
            self._play_audio_file_sync(file_path)

    def generate_speech_file(self, text: str, save_dir: Optional[str] = None) -> Optional[str]:
        if not text or not text.strip():
            return None

        if save_dir is None:
            save_dir = tempfile.gettempdir()
        os.makedirs(save_dir, exist_ok=True)

        language = self._detect_tts_language(text)
        print(f"Detected language: {language}")

        if self.prefer_moss_tts:
            try:
                return self._moss_generate_speech_file(text, language, save_dir)
            except Exception as exc:
                print(f"MOSS-TTS-Nano synthesis failed, trying Edge/gTTS fallback: {exc!r}")

        if self.prefer_edge_tts:
            try:
                return self._edge_generate_speech_file(text, language, save_dir)
            except Exception as exc:
                print(f"Edge TTS synthesis failed, trying gTTS fallback: {exc}")

        if self.use_gtts:
            try:
                return self._gtts_generate_speech_file(text, language, save_dir)
            except Exception as exc:
                print(f"gTTS synthesis failed: {exc}")

        print("TTS synthesis failed: no engine produced audio")
        return None

    def _moss_configured(self) -> bool:
        if importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None:
            if os.path.exists(os.path.join(self.moss_repo_path, "moss_tts_nano_runtime.py")):
                return True
            if os.path.exists(self.moss_model_path) or os.environ.get("MOSS_TTS_MODEL_NAME"):
                return True
        if os.environ.get("ENABLE_MOSS_TTS_PROCESS_FALLBACK", "0") == "1" and self._moss_process_configured():
            return True
        return False

    def _moss_process_configured(self) -> bool:
        if self.moss_command:
            return True
        if self.moss_cli:
            return True
        return bool(self.moss_python and os.path.exists(os.path.join(self.moss_repo_path, "infer.py")))

    def _moss_runtime_label(self) -> str:
        if self._moss_runtime == "direct":
            if os.path.exists(os.path.join(self.moss_repo_path, "moss_tts_nano_runtime.py")):
                return os.path.join(self.moss_repo_path, "moss_tts_nano_runtime.py")
            return self._moss_model_source()
        if self.moss_command:
            return "MOSS_TTS_COMMAND"
        if self.moss_cli:
            return self.moss_cli
        return os.path.join(self.moss_repo_path, "infer.py")

    def _moss_generate_speech_file(self, text: str, language: str, save_dir: str) -> str:
        if self._moss_runtime == "direct":
            audio_path = self._moss_generate_direct_speech_file(text, language, save_dir)
        else:
            audio_path = self._moss_generate_process_speech_file(text, language, save_dir)
        return self._maybe_speedup_audio_file(audio_path)

    def _moss_model_source(self) -> str:
        return self.moss_model_path if os.path.exists(self.moss_model_path) else self.moss_model_name

    def _load_moss_direct(self) -> None:
        if self._moss_model is not None and self._moss_processor is not None:
            return

        if os.path.exists(os.path.join(self.moss_repo_path, "moss_tts_nano_runtime.py")):
            self._load_moss_official_service()
            return

        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor
        if hasattr(torch.backends, "cuda"):
            try:
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

        device = self.moss_device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model_source = self._moss_model_source()
        attn_implementation = self._resolve_moss_attn_implementation(torch, device, dtype)
        local_files_only = self.moss_local_files_only

        print(
            "Loading MOSS-TTS-Nano direct model: "
            f"source={model_source}, device={device}, dtype={dtype}, attn={attn_implementation}"
        )
        self._moss_processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self._moss_model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        self._moss_model.eval()
        self.moss_device = device
        self._moss_torch = torch
        self._moss_torchaudio = torchaudio
        print("MOSS-TTS-Nano direct model loaded")

    def _load_moss_official_service(self) -> None:
        if self._moss_service is not None:
            return

        if self.moss_repo_path not in sys.path:
            sys.path.insert(0, self.moss_repo_path)

        from moss_tts_nano_runtime import NanoTTSService

        checkpoint_path = self.moss_model_path
        if self.moss_local_files_only and not os.path.exists(checkpoint_path):
            local_candidate = os.path.join(self.moss_repo_path, "checkpoint")
            if os.path.exists(local_candidate):
                checkpoint_path = local_candidate
            else:
                raise FileNotFoundError(
                    "MOSS-TTS-Nano checkpoint is not available locally. "
                    "Set MOSS_TTS_MODEL_PATH to the downloaded checkpoint directory, "
                    "or set MOSS_TTS_LOCAL_FILES_ONLY=0 to allow online loading."
                )
        if self.moss_local_files_only and not os.path.exists(self.moss_audio_tokenizer_path):
            local_audio_tokenizer = os.path.join(self.moss_repo_path, "audio_tokenizer")
            if os.path.exists(local_audio_tokenizer):
                self.moss_audio_tokenizer_path = local_audio_tokenizer
            else:
                raise FileNotFoundError(
                    "MOSS-Audio-Tokenizer-Nano checkpoint is not available locally. "
                    "Set MOSS_TTS_AUDIO_TOKENIZER_PATH to the downloaded tokenizer directory, "
                    "or set MOSS_TTS_LOCAL_FILES_ONLY=0 to allow online loading."
                )

        print(
            "Loading MOSS-TTS-Nano official service: "
            f"checkpoint={checkpoint_path}, audio_tokenizer={self.moss_audio_tokenizer_path}, "
            f"device={self.moss_device or 'auto'}, dtype={self.moss_dtype}, attn={self.moss_attn_implementation}"
        )
        self._moss_service = NanoTTSService(
            checkpoint_path=checkpoint_path,
            audio_tokenizer_path=self.moss_audio_tokenizer_path,
            device=self.moss_device or "auto",
            dtype=self.moss_dtype,
            attn_implementation=self.moss_attn_implementation,
            output_dir=os.path.join(PROJECT_ROOT, "code", "temp_audio"),
        )
        preload_voices = sorted({self.moss_voice, self.moss_zh_voice, self.moss_en_voice})
        preload_info = self._moss_service.preload(voices=preload_voices, load_model=True)
        print(
            "MOSS-TTS-Nano official service loaded: "
            f"device={preload_info.get('device')}, dtype={preload_info.get('dtype')}, "
            f"attn={preload_info.get('configured_attn_implementation')}, "
            f"codec_attn={preload_info.get('configured_codec_attn_implementation')}, "
            f"voice={self.moss_voice}, zh_voice={self.moss_zh_voice}, en_voice={self.moss_en_voice}"
        )

    @staticmethod
    def _resolve_moss_attn_implementation(torch_module, device: str, dtype) -> str:
        if (
            device.startswith("cuda")
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch_module.float16, torch_module.bfloat16}
        ):
            try:
                major, _ = torch_module.cuda.get_device_capability()
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        if device.startswith("cuda"):
            return "sdpa"
        return "eager"

    def _moss_generate_direct_speech_file(self, text: str, language: str, save_dir: str) -> str:
        self._load_moss_direct()
        if self._moss_service is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=save_dir)
            output_path = temp_file.name
            temp_file.close()
            started_at = time.time()
            voice = self._moss_voice_for_text(text, language)
            max_new_frames = self._moss_max_frames_for_text(text, language)
            result = self._moss_service.synthesize(
                text=text,
                voice=voice,
                mode=self.moss_mode,
                output_audio_path=output_path,
                prompt_audio_path=self.moss_reference_audio or None,
                max_new_frames=max_new_frames,
                voice_clone_max_text_tokens=self.moss_voice_clone_max_text_tokens,
                audio_temperature=self.moss_audio_temperature,
                attn_implementation=self.moss_attn_implementation,
            )
            audio_path = str(result.get("audio_path") or output_path)
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise RuntimeError("MOSS-TTS-Nano official service produced no audio")
            elapsed = float(result.get("elapsed_seconds") or (time.time() - started_at))
            sample_rate = int(result.get("sample_rate") or 0)
            waveform = result.get("waveform")
            duration = 0.0
            try:
                if waveform is not None and sample_rate > 0:
                    duration = float(waveform.shape[-1]) / sample_rate
            except Exception:
                duration = 0.0
            rtf = elapsed / duration if duration > 0 else 0.0
            print(
                "[OK] MOSS-TTS-Nano direct speech file generated: "
                f"{audio_path}, elapsed={elapsed:.2f}s, audio={duration:.2f}s, rtf={rtf:.2f}, "
                f"device={getattr(self._moss_service, 'device', 'unknown')}, "
                f"dtype={getattr(self._moss_service, 'dtype', 'unknown')}, "
                f"voice={voice}, max_frames={max_new_frames}"
            )
            return audio_path

        if self._moss_model is None or self._moss_processor is None:
            raise RuntimeError("MOSS-TTS-Nano direct model is not loaded")

        torch = self._moss_torch
        torchaudio = self._moss_torchaudio
        if torch is None or torchaudio is None:
            raise RuntimeError("MOSS-TTS-Nano runtime libraries are not loaded")

        conversation = [self._moss_processor.build_user_message(text=text)]
        if self.moss_reference_audio:
            conversation = [
                self._moss_processor.build_user_message(text=text, reference=[self.moss_reference_audio])
            ]

        batch = self._moss_processor([conversation], mode="generation")
        input_ids = batch["input_ids"].to(self.moss_device)
        attention_mask = batch["attention_mask"].to(self.moss_device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.moss_max_new_frames,
        }
        if self.moss_audio_temperature > 0:
            generate_kwargs["audio_temperature"] = self.moss_audio_temperature

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=save_dir)
        output_path = temp_file.name
        temp_file.close()

        with torch.no_grad():
            outputs = self._moss_model.generate(**generate_kwargs)

        decoded_messages = self._moss_processor.decode(outputs)
        for message in decoded_messages:
            audio = message.audio_codes_list[0]
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            sampling_rate = int(self._moss_processor.model_config.sampling_rate)
            torchaudio.save(output_path, audio.detach().cpu(), sampling_rate)
            print(f"[OK] MOSS-TTS-Nano direct speech file generated: {output_path}")
            return output_path

        raise RuntimeError("MOSS-TTS-Nano direct generation returned no audio")

    def _moss_voice_for_text(self, text: str, language: str) -> str:
        return self.moss_voice

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _detect_tts_language(self, text: str) -> str:
        """Classify mixed TTS text conservatively for stable voice selection."""
        raw = text or ""
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", raw))
        latin_words = len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", raw))
        if cjk_count > 0 and latin_words > 0:
            return "mixed"
        return LanguageUtils.detect_text_language(raw)

    def _moss_max_frames_for_text(self, text: str, language: str) -> int:
        """Bound generation length so short prompts cannot drift into long noise."""
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_words = len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text))

        if language == "en":
            estimated_seconds = max(1.2, latin_words * 0.42 + 0.8)
        elif language == "zh":
            estimated_seconds = max(1.0, cjk_chars * 0.22 + 0.8)
        else:
            estimated_seconds = max(1.2, cjk_chars * 0.20 + latin_words * 0.38 + 0.9)

        frame_rate = 12.5
        safety = float(os.environ.get("MOSS_TTS_FRAME_SAFETY", "1.45"))
        min_frames = int(os.environ.get("MOSS_TTS_MIN_NEW_FRAMES", "24"))
        dynamic_frames = int(estimated_seconds * frame_rate * safety)
        return max(min_frames, min(self.moss_max_new_frames, dynamic_frames))

    def _moss_generate_process_speech_file(self, text: str, language: str, save_dir: str) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=save_dir)
        output_path = temp_file.name
        temp_file.close()

        commands = self._moss_candidate_commands(text, language, output_path)
        errors = []
        for command, shell in commands:
            try:
                self._run_moss_command(command, shell=shell)
                resolved_output = self._resolve_moss_output(output_path, save_dir)
                print(f"[OK] MOSS-TTS-Nano speech file generated: {resolved_output}")
                return resolved_output
            except Exception as exc:
                errors.append(f"{command!r}: {exc}")

        raise RuntimeError("All MOSS-TTS-Nano commands failed. " + " | ".join(errors[-3:]))

    def _maybe_speedup_audio_file(self, file_path: str) -> str:
        """Apply a small post-generation tempo boost to reduce sluggish TTS playback."""
        speed = self.moss_speed
        if speed <= 0 or abs(speed - 1.0) < 0.01:
            return file_path

        try:
            import numpy as np
            import soundfile as sf
            from scipy import signal

            audio_data, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
            if audio_data.size == 0:
                return file_path

            original_samples = int(audio_data.shape[0])
            target_samples = max(1, int(round(original_samples / speed)))
            if target_samples == original_samples:
                return file_path

            sped_audio = signal.resample(audio_data, target_samples, axis=0).astype("float32")
            sped_audio = np.clip(sped_audio, -1.0, 1.0)
            sf.write(file_path, sped_audio, sample_rate)

            original_duration = original_samples / float(sample_rate)
            new_duration = target_samples / float(sample_rate)
            print(
                "[OK] MOSS-TTS-Nano audio speedup applied: "
                f"speed={speed:.2f}x, duration={original_duration:.2f}s->{new_duration:.2f}s"
            )
        except Exception as exc:
            print(f"MOSS-TTS-Nano audio speedup skipped: {exc!r}")
        return file_path

    def _moss_candidate_commands(self, text: str, language: str, output_path: str) -> list[tuple[list[str] | str, bool]]:
        format_values = {
            "text": text,
            "output": output_path,
            "language": "zh" if language in {"zh", "mixed"} else "en",
            "repo": self.moss_repo_path,
            "model": self.moss_model_path,
            "voice": self.moss_voice,
            "reference": self.moss_reference_audio,
        }

        if self.moss_command:
            return [(self.moss_command.format(**format_values), True)]

        commands: list[tuple[list[str] | str, bool]] = []
        if self.moss_cli:
            base = [self.moss_cli, "generate", "--text", text, "--output", output_path]
            commands.append((self._append_moss_optional_args(base, language), False))
            base = [self.moss_cli, "--text", text, "--output", output_path]
            commands.append((self._append_moss_optional_args(base, language), False))

        infer_py = os.path.join(self.moss_repo_path, "infer.py")
        if self.moss_python and os.path.exists(infer_py):
            base = [self.moss_python, infer_py, "--text", text, "--output", output_path]
            commands.append((self._append_moss_optional_args(base, language), False))
            base = [self.moss_python, infer_py, "--text", text, "--output_path", output_path]
            commands.append((self._append_moss_optional_args(base, language), False))
            base = [self.moss_python, infer_py, "--text", text, "--out_path", output_path]
            commands.append((self._append_moss_optional_args(base, language), False))

        return commands

    def _append_moss_optional_args(self, command: list[str], language: str) -> list[str]:
        language_code = "zh" if language in {"zh", "mixed"} else "en"
        result = list(command)
        if os.path.exists(self.moss_model_path):
            result.extend(["--model_path", self.moss_model_path])
        if self.moss_voice:
            result.extend(["--voice", self.moss_voice])
        if self.moss_reference_audio:
            result.extend(["--reference_audio", self.moss_reference_audio])
        result.extend(["--language", language_code])
        return result

    def _run_moss_command(self, command: list[str] | str, shell: bool = False) -> None:
        env = os.environ.copy()
        env.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))
        env.setdefault("MOSS_TTS_MODEL_PATH", self.moss_model_path)
        process = subprocess.run(
            command,
            cwd=self.moss_repo_path if os.path.isdir(self.moss_repo_path) else PROJECT_ROOT,
            env=env,
            shell=shell,
            text=True,
            capture_output=True,
            timeout=self.moss_timeout,
        )
        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            raise RuntimeError(stderr or stdout or f"exit code {process.returncode}")

    @staticmethod
    def _resolve_moss_output(output_path: str, save_dir: str) -> str:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        candidates = []
        for file_name in os.listdir(save_dir):
            if file_name.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
                file_path = os.path.join(save_dir, file_name)
                candidates.append((os.path.getmtime(file_path), file_path))
        if candidates:
            newest = max(candidates)[1]
            if os.path.getsize(newest) > 0:
                return newest

        raise RuntimeError("MOSS-TTS-Nano command completed but no audio file was produced")

    def _edge_generate_speech_file(self, text: str, language: str, save_dir: str) -> str:
        voice = "zh-CN-XiaoxiaoNeural" if language in {"zh", "mixed"} else "en-US-AriaNeural"

        async def generate_speech():
            communicate = edge_tts.Communicate(text, voice)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=save_dir)
            temp_file_path = temp_file.name
            temp_file.close()
            await communicate.save(temp_file_path)
            return temp_file_path

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            temp_file_path = loop.run_until_complete(generate_speech())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        print(f"[OK] Edge TTS speech file generated: {temp_file_path}")
        return temp_file_path

    def _gtts_generate_speech_file(self, text: str, language: str, save_dir: str) -> str:
        tld = "com.cn" if language == "zh" else "com"
        lang = "en" if language in {"en", "mixed"} else "zh-CN"
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=save_dir)
        temp_file_path = temp_file.name
        temp_file.close()
        tts.save(temp_file_path)
        print(f"[OK] gTTS speech file generated: {temp_file_path}")
        return temp_file_path

    def _play_audio_file(self, file_path: str) -> None:
        if not PYGAME_AVAILABLE:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            self._safe_unlink(file_path)

    def _play_audio_file_sync(self, file_path: str) -> None:
        if not PYGAME_AVAILABLE:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.stop()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            time.sleep(0.1)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            self._safe_unlink(file_path)

    @staticmethod
    def _safe_unlink(file_path: str) -> None:
        try:
            os.unlink(file_path)
        except OSError:
            pass
