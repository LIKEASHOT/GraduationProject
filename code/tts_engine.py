#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-to-speech engine with CosyVoice as the preferred backend.
Falls back to Edge TTS and gTTS only when CosyVoice is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf

from language_utils import LanguageUtils

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

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_cosyvoice_repo = os.environ.get("COSYVOICE_REPO_PATH") or os.path.join(project_root, "CosyVoice")
    if os.path.exists(local_cosyvoice_repo) and local_cosyvoice_repo not in sys.path:
        sys.path.insert(0, local_cosyvoice_repo)
    local_matcha = os.path.join(local_cosyvoice_repo, "third_party", "Matcha-TTS")
    if os.path.exists(local_matcha) and local_matcha not in sys.path:
        sys.path.insert(0, local_matcha)
    from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel
    COSYVOICE_AVAILABLE = True
except ImportError:
    try:
        from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceModel
        COSYVOICE_AVAILABLE = True
    except ImportError:
        COSYVOICE_AVAILABLE = False
        CosyVoiceModel = None

try:
    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA设备数量: {torch.cuda.device_count()}")
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        print("❌ CUDA不可用，将使用CPU")
except ImportError:
    print("⚠️ PyTorch未安装，跳过CUDA检查")


class TTSEngine:
    """TTS engine that prefers CosyVoice for both HTTP and realtime paths."""

    def __init__(self, prefer_edge_tts=True, prefer_local_tts=False):
        self.user_prefer_edge_tts = prefer_edge_tts
        self.user_prefer_local_tts = prefer_local_tts
        self.use_gtts = GTTS_AVAILABLE

        self.prefer_cosyvoice = False
        self.prefer_edge_tts = False
        self.current_engine: Optional[str] = None

        self.cosyvoice_model = None
        self.cosyvoice_model_path = os.environ.get("COSYVOICE_MODEL_PATH")
        self.cosyvoice_speaker = os.environ.get("COSYVOICE_SPEAKER")

    def init_tts(self):
        """Initialize TTS backends with CosyVoice first."""
        cosyvoice_available = self._check_cosyvoice_available()
        edge_tts_available = self._check_edge_tts_available()

        self.prefer_cosyvoice = cosyvoice_available
        self.prefer_edge_tts = edge_tts_available and self.user_prefer_edge_tts

        print(
            f"TTS引擎配置 - CosyVoice优先: {self.prefer_cosyvoice}, "
            f"Edge备选: {self.prefer_edge_tts}, gTTS备选: {self.use_gtts}"
        )

        if self.prefer_cosyvoice:
            self._init_cosyvoice()
            self.current_engine = "cosyvoice"
            print("✅ 使用CosyVoice引擎")
            return True

        if self.prefer_edge_tts:
            self.current_engine = "edge"
            print("✅ 使用Edge TTS引擎")
            return True

        if self.use_gtts:
            self.current_engine = "gtts"
            print("✅ 使用gTTS引擎（Google语音合成）")
            return True

        print("❌ 没有可用的TTS引擎")
        return False

    def _check_cosyvoice_available(self):
        if not COSYVOICE_AVAILABLE:
            return False
        return self._resolve_cosyvoice_model_path() is not None

    @staticmethod
    def _check_edge_tts_available():
        return EDGE_TTS_AVAILABLE

    @staticmethod
    def _check_audio_device():
        if not PYGAME_AVAILABLE:
            return False
        try:
            pygame.mixer.init()
            pygame.mixer.quit()
            return True
        except Exception:
            return False

    def get_current_engine_info(self):
        engine_map = {
            "cosyvoice": "FunAudioLLM CosyVoice",
            "edge": "Microsoft Edge TTS",
            "gtts": "Google gTTS",
        }
        return engine_map.get(self.current_engine, "未知引擎")

    def _resolve_cosyvoice_model_path(self):
        candidates = []
        if self.cosyvoice_model_path:
            candidates.append(self.cosyvoice_model_path)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.extend(
            [
                os.path.join(project_root, "models", "CosyVoice-300M-SFT"),
                os.path.join(project_root, "models", "CosyVoice-300M-Instruct"),
                os.path.join(project_root, "models", "CosyVoice-300M"),
                os.path.join(project_root, "models", "CosyVoice2-0.5B"),
            ]
        )

        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _init_cosyvoice(self):
        if self.cosyvoice_model is not None:
            return self.cosyvoice_model

        model_path = self._resolve_cosyvoice_model_path()
        if not model_path:
            raise RuntimeError(
                "CosyVoice model directory not found. Set COSYVOICE_MODEL_PATH or place the model under models/."
            )

        use_fp16 = self._should_use_fp16()
        self.cosyvoice_model = CosyVoiceModel(model_path, fp16=use_fp16)
        self.cosyvoice_speaker = self._resolve_cosyvoice_speaker()
        print(f"CosyVoice TTS ready: {model_path}")
        print(f"CosyVoice speaker: {self.cosyvoice_speaker}")
        print(f"CosyVoice fp16: {use_fp16}")
        return self.cosyvoice_model

    def _resolve_cosyvoice_speaker(self):
        requested = os.environ.get("COSYVOICE_SPEAKER")
        available = []
        if hasattr(self.cosyvoice_model, "list_available_spks"):
            try:
                available = list(self.cosyvoice_model.list_available_spks())
            except Exception:
                available = []

        if requested:
            if not available or requested in available:
                return requested
            print(f"Requested CosyVoice speaker `{requested}` not found. Available speakers: {available}")

        if available:
            return available[0]

        raise RuntimeError(
            "CosyVoice model has no SFT speakers. Use a model with `spk2info.pt` "
            "or set up zero-shot/cross-lingual inference instead of inference_sft."
        )

    def text_to_speech(self, text, async_play=True, language=None):
        """Generate speech and optionally play it back."""
        if language is None:
            language = LanguageUtils.detect_text_language(text)

        file_path = self.generate_speech_file(text)
        if not file_path:
            return

        if async_play:
            play_thread = threading.Thread(target=self._play_audio_file, args=(file_path,), daemon=True)
            play_thread.start()
        else:
            self._play_audio_file_sync(file_path)

    def generate_speech_file(self, text, save_dir=None):
        """Generate a speech file and return its path."""
        if not text or not text.strip():
            return None

        if save_dir is None:
            save_dir = tempfile.gettempdir()
        os.makedirs(save_dir, exist_ok=True)

        language = LanguageUtils.detect_text_language(text)
        print(f"检测到语言: {language}")

        try:
            if self.prefer_cosyvoice:
                return self._cosyvoice_generate_speech_file(text, save_dir)
            if self.prefer_edge_tts:
                return self._edge_generate_speech_file(text, language, save_dir)
            if self.use_gtts:
                return self._gtts_generate_speech_file(text, language, save_dir)
            print("❌ 没有可用的TTS引擎进行语音合成")
            return None
        except Exception as exc:
            print(f"语音文件生成失败: {exc}")
            return None

    def _cosyvoice_generate_speech_file(self, text, save_dir):
        self._init_cosyvoice()
        generator = self.cosyvoice_model.inference_sft(text, self.cosyvoice_speaker, stream=True)

        for item in generator:
            waveform = item.get("tts_speech")
            if waveform is None:
                continue

            audio = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
            audio = np.squeeze(audio).astype(np.float32)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=save_dir)
            temp_file_path = temp_file.name
            temp_file.close()

            sf.write(temp_file_path, audio, getattr(self.cosyvoice_model, "sample_rate", 22050))
            print(f"[OK] CosyVoice语音文件生成完成: {temp_file_path}")
            return temp_file_path

        raise RuntimeError("CosyVoice returned no audio")

    @staticmethod
    def _should_use_fp16():
        if os.environ.get("COSYVOICE_FP16", "1") == "0":
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _edge_generate_speech_file(self, text, language, save_dir):
        if language == "zh":
            voice = "zh-CN-XiaoxiaoNeural"
        else:
            voice = "en-US-AriaNeural"

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

        print(f"[OK] Edge语音文件生成完成: {temp_file_path}")
        return temp_file_path

    def _gtts_generate_speech_file(self, text, language, save_dir):
        tld = "com.cn" if language == "zh" else "com"
        tts = gTTS(text=text, lang="en" if language in ["en", "mixed"] else "zh-cn", tld=tld, slow=False)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=save_dir)
        temp_file_path = temp_file.name
        temp_file.close()
        tts.save(temp_file_path)
        print(f"[OK] gTTS语音文件生成完成: {temp_file_path}")
        return temp_file_path

    def _play_audio_file(self, file_path):
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
            try:
                os.unlink(file_path)
            except OSError:
                pass

    def _play_audio_file_sync(self, file_path):
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
            try:
                os.unlink(file_path)
            except OSError:
                pass
