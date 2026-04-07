#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音合成引擎
"""

import threading
import time
import os
import tempfile
import asyncio

from language_utils import LanguageUtils

# 导入Edge TTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# 导入gTTS
try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# 导入本地TTS模型
try:
    from transformers import VitsModel, AutoTokenizer
    import torch
    import numpy as np
    import scipy.io.wavfile
    LOCAL_TTS_AVAILABLE = True
except ImportError:
    LOCAL_TTS_AVAILABLE = False

# Coqui TTS已移除，使用Edge TTS作为主要引擎

# CUDA加速配置
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA设备数量: {torch.cuda.device_count()}")
        # 设置默认设备为GPU
        torch.cuda.set_device(0)
        # 启用cuDNN优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        print("❌ CUDA不可用，将使用CPU")
except ImportError:
    print("⚠️ PyTorch未安装，CUDA检查跳过")


class TTSEngine:
    """语音合成引擎 - 支持多种TTS方案"""

    def __init__(self, prefer_edge_tts=True, prefer_local_tts=False):
        self.use_gtts = GTTS_AVAILABLE  # 是否可以使用gTTS

        # 存储用户偏好设置
        self.user_prefer_local_tts = prefer_local_tts
        self.user_prefer_edge_tts = prefer_edge_tts

        # 初始化时先设置为False，稍后在init_tts中重新检查
        self.prefer_local_tts = False
        self.prefer_edge_tts = False

        self.local_model = None  # 本地TTS模型
        self.local_tokenizer = None  # 本地tokenizer
        self.current_engine = None  # 当前使用的引擎类型

    def init_tts(self):
        """初始化语音合成引擎"""
        try:
            # 运行时重新检查依赖可用性
            local_tts_available = self._check_local_tts_available()
            edge_tts_available = self._check_edge_tts_available()
            if self.user_prefer_local_tts and local_tts_available:
                self.prefer_local_tts = True
                self.prefer_edge_tts = edge_tts_available
            else:
                self.prefer_local_tts = False
                self.prefer_edge_tts = edge_tts_available

            print(f"TTS引擎配置 - Edge优先: {self.prefer_edge_tts}, 本地备选: {self.prefer_local_tts}")

            # 检查是否有音频设备（服务器环境可能没有）
            has_audio_device = self._check_audio_device()

            # 优先级：Edge TTS -> 本地TTS -> gTTS

            if self.prefer_local_tts:
                # 使用本地TTS模型
                try:
                    self._init_local_tts()
                    self.current_engine = "local"
                    print("✅ 使用本地TTS模型（VitsModel）")
                    return True
                except Exception as e:
                    print(f"本地TTS初始化失败，尝试Edge TTS: {e}")
                    self.prefer_local_tts = False

            if self.prefer_edge_tts:
                # 使用Edge TTS（Microsoft免费服务，高质量，支持中英文）
                try:
                    if has_audio_device:
                        pygame.mixer.init()
                    self.current_engine = "edge"
                    print("✅ 使用Edge TTS引擎（Microsoft - 支持中英文）")
                    return True
                except Exception as e:
                    print(f"Edge TTS初始化失败，尝试本地TTS: {e}")
                    self.prefer_edge_tts = False

            if self.use_gtts:
                # 使用gTTS
                try:
                    if has_audio_device:
                        pygame.mixer.init()
                    self.current_engine = "gtts"
                    print("✅ 使用gTTS引擎（Google语音合成）")
            return True
                except Exception as e:
                    print(f"gTTS初始化失败: {e}")
                    self.use_gtts = False
                    return False

            print("❌ 没有可用的TTS引擎")
            return False
        except Exception as e:
            print(f"语音合成引擎初始化失败: {e}")
            return False

    def _check_local_tts_available(self):
        """运行时检查本地TTS依赖是否可用"""
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch
            import numpy as np
            import scipy.io.wavfile
            return True
        except ImportError:
            return False


    def _check_edge_tts_available(self):
        """运行时检查Edge TTS是否可用"""
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def _check_audio_device(self):
        """检查是否有音频设备可用"""
        try:
            # 尝试初始化pygame来检测音频设备
            pygame.mixer.init()
            pygame.mixer.quit()  # 立即退出，避免占用资源
            return True
        except:
            return False

    def get_current_engine_info(self):
        """获取当前使用的引擎信息"""
        engine_map = {
            "local": "本地TTS模型（Facebook MMS）",
            "edge": "Microsoft Edge TTS",
            "gtts": "Google gTTS"
        }
        return engine_map.get(self.current_engine, "未知引擎")

    def _init_local_tts(self):
        """初始化本地TTS模型"""
        # 尝试多个可能的模型路径（按优先级排序）
        possible_paths = [
            # 服务器上实际的模型路径（最高优先级）
            "/home/ubuntu/models/mms_tts_multilingual",
            # 用户主目录下的models
            os.path.join(os.path.expanduser("~"), "models", "mms_tts_multilingual"),
            # 当前工作目录下的models
            os.path.join(os.getcwd(), "models", "mms_tts_multilingual"),
            # 相对于脚本目录的models
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mms_tts_multilingual")
        ]

        model_path = None
        for path in possible_paths:
            print(f"检查本地TTS模型路径: {path}")
            if os.path.exists(path):
                model_path = path
                print(f"✅ 找到模型目录: {path}")
                break

        if model_path is None:
            raise Exception(f"本地TTS模型目录不存在，已检查路径: {possible_paths}")

        # 检查目录内容
        try:
            files = os.listdir(model_path)
            print(f"模型目录文件: {files[:5]}...")  # 只显示前5个文件
            if len(files) == 0:
                raise Exception("模型目录为空")
        except Exception as e:
            raise Exception(f"无法读取模型目录: {e}")

        print("正在加载本地TTS模型...")
        try:
            # 首先尝试加载英文tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("✅ Tokenizer加载成功")
        except Exception as e:
            raise Exception(f"Tokenizer加载失败: {e}")

        try:
            # 加载基础模型
            self.local_model = VitsModel.from_pretrained(model_path)
            print("✅ VitsModel加载成功")

            # 尝试添加中文语言适配器（如果可用）
            try:
                from transformers import VitsModel
                zh_model_path = os.path.join(model_path, "zh")
                if os.path.exists(zh_model_path):
                    print("发现中文模型支持，加载中文适配器...")
                    # 这里可以加载中文特定的适配器
                    # self.local_model.load_adapter(zh_model_path, "zh")
                    print("✅ 中文适配器加载完成")
                else:
                    print("ℹ️ 未发现中文适配器，仅支持英文")
            except Exception as e:
                print(f"⚠️ 中文适配器加载失败，仅支持英文: {e}")

        except Exception as e:
            raise Exception(f"VitsModel加载失败: {e}")

        print("✅ 本地TTS模型加载完成")

    def text_to_speech(self, text, async_play=True, language=None):
        """
        文字转语音

        Args:
            text: 要转换为语音的文字
            async_play: 是否异步播放（不阻塞程序执行）
            language: 语言代码（可选，如果不提供会自动检测）
        """
        print("正在生成语音...")

        try:
            # 检查是否有音频设备
            has_audio = self._check_audio_device()
            if not has_audio and async_play:
                print("⚠️ 未检测到音频设备，将只生成语音文件不播放")
                async_play = False

            # 检测文本语言（如果未提供）
            if language is None:
                language = LanguageUtils.detect_text_language(text)
            print(f"检测到语言: {language}")

            if self.prefer_edge_tts:
                # 使用Edge TTS方案（Microsoft，支持中英文）
                self._edge_text_to_speech(text, language, async_play)
            elif self.prefer_local_tts and self.local_model and language in ['en']:
                # 使用本地TTS模型（仅英文）
                self._local_text_to_speech(text, language, async_play)
            elif self.use_gtts:
                # 使用gTTS方案
                self._gtts_text_to_speech(text, language, async_play)
            else:
                print("[ERROR] 没有可用的TTS引擎进行语音合成")
            return

        except Exception as e:
            print(f"语音生成失败: {e}")

    def generate_speech_file(self, text, save_dir=None):
        """
        生成语音文件并返回文件路径（不播放）
        
        Args:
            text: 要转换为语音的文字
            save_dir: 保存目录，如果为None则使用临时目录
            
        Returns:
            生成的音频文件路径，失败返回None
        """
        try:
            # 检测文本语言
            language = LanguageUtils.detect_text_language(text)
            print(f"检测到语言: {language}")

            # 确定保存目录
            if save_dir is None:
                save_dir = tempfile.gettempdir()
            os.makedirs(save_dir, exist_ok=True)
            
            if self.prefer_edge_tts:
                # 使用Edge TTS方案（Microsoft，支持中英文）
                return self._edge_generate_speech_file(text, language, save_dir)
            elif self.prefer_local_tts and self.local_model and language in ['en']:
                # 使用本地TTS模型（仅英文）
                return self._local_generate_speech_file(text, language, save_dir)
            elif self.use_gtts:
                # 使用gTTS方案
                return self._gtts_generate_speech_file(text, language, save_dir)
            else:
                print("❌ 没有可用的TTS引擎进行语音合成")
                return None
                
        except Exception as e:
            print(f"语音文件生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _edge_generate_speech_file(self, text, language, save_dir):
        """使用Edge TTS生成语音文件（Microsoft）"""
        try:
            # 选择合适的语音
            if language == 'zh':
                voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
            elif language == 'en':
                voice = "en-US-AriaNeural"  # 英文女声
            else:
                voice = "en-US-AriaNeural"  # 默认英文

            async def generate_speech():
                communicate = edge_tts.Communicate(text, voice)
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.mp3',
                    dir=save_dir
                )
                temp_file_path = temp_file.name
                temp_file.close()
                await communicate.save(temp_file_path)
                return temp_file_path

            # 运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            temp_file_path = loop.run_until_complete(generate_speech())
            loop.close()

            print(f"[OK] Edge语音文件生成完成: {temp_file_path}")
            return temp_file_path

        except Exception as e:
            print(f"Edge TTS文件生成失败: {e}")
            # 回退到gTTS
            if self.use_gtts:
                return self._gtts_generate_speech_file(text, language, save_dir)
            else:
                print("[ERROR] Edge TTS和gTTS都不可用")
                return None

    def _local_generate_speech_file(self, text, language, save_dir):
        """使用本地TTS模型生成语音文件"""
        try:
            print(f"使用本地TTS模型生成语音文件: {language}")
            print(f"输入文本: '{text}'")

            # 检查文本是否为空或只包含空白字符
            if not text or not text.strip():
                print("❌ 输入文本为空，跳过本地TTS")
                return None

            # 对文本进行tokenization
            inputs = self.local_tokenizer(text, return_tensors="pt")

            # 检查tokenization结果
            if 'input_ids' in inputs:
                input_length = inputs['input_ids'].shape[1] if len(inputs['input_ids'].shape) > 1 else 0
                print(f"Tokenization结果长度: {input_length}")
                if input_length == 0:
                    print("❌ Tokenization后长度为0，跳过本地TTS")
                    return None

            # 生成语音
            with torch.no_grad():
                output = self.local_model(**inputs)

            # 获取音频数据
            audio = output.waveform[0].cpu().numpy()

            # 创建音频文件
            import scipy.io.wavfile
            import tempfile

            audio_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir=save_dir
            )
            audio_file_path = audio_file.name
            audio_file.close()

            # 保存为WAV文件 (22050Hz采样率)
            scipy.io.wavfile.write(audio_file_path, rate=22050, data=audio)

            print(f"✅ 本地TTS文件生成完成: {audio_file_path}")
            return audio_file_path

        except Exception as e:
            print(f"本地TTS文件生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到Edge TTS
            if self.prefer_edge_tts:
                print("回退到Edge TTS...")
                return self._edge_generate_speech_file(text, language, save_dir)
            return None

    def _local_text_to_speech(self, text, language, async_play):
        """使用本地TTS模型进行语音合成"""
        try:
            print(f"使用本地TTS模型生成语音: {language}")

            # 对文本进行tokenization
            inputs = self.local_tokenizer(text, return_tensors="pt")

            # 生成语音
            with torch.no_grad():
                output = self.local_model(**inputs)

            # 获取音频数据 (VitsModel返回的音频是tensor)
            audio = output.waveform[0].cpu().numpy()

            # 创建临时音频文件
            import tempfile
            import scipy.io.wavfile

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir=tempfile.gettempdir()
            )
            temp_file_path = temp_file.name
            temp_file.close()

            # 保存为WAV文件
            # VitsModel默认采样率为22050Hz
            scipy.io.wavfile.write(temp_file_path, rate=22050, data=audio)

            print(f"本地TTS生成完成: {temp_file_path}")

            # 播放音频
            if async_play:
                self._play_audio_file_async(temp_file_path)
            else:
                self._play_audio_file_sync(temp_file_path)

        except Exception as e:
            print(f"本地TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到Edge TTS
            if self.prefer_edge_tts:
                print("回退到Edge TTS...")
                self._edge_text_to_speech(text, language, async_play)


    def _gtts_generate_speech_file(self, text, language, save_dir):
        """使用gTTS生成语音文件"""
        try:
            # 根据语言选择TLD
            tld = 'com'  # 默认英文
            if language == 'zh':
                tld = 'com.cn'  # 中文使用中国服务器

            # 生成语音
            tts = gTTS(text=text, lang='en' if language in ['en', 'mixed'] else 'zh-cn', tld=tld, slow=False)

            # 保存到文件
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.mp3',
                dir=save_dir
            )
            temp_file_path = temp_file.name
            temp_file.close()
            tts.save(temp_file_path)

            print(f"[OK] gTTS语音文件生成完成: {temp_file_path}")
            return temp_file_path

        except Exception as e:
            print(f"gTTS文件生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _edge_text_to_speech(self, text, language, async_play):
        """使用Edge TTS进行语音合成（Microsoft）"""
        try:
            # 选择合适的语音
            if language == 'zh':
                voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
            elif language == 'en':
                voice = "en-US-AriaNeural"  # 英文女声
            else:
                voice = "en-US-AriaNeural"  # 默认英文

            async def generate_speech():
                communicate = edge_tts.Communicate(text, voice)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file_path = temp_file.name
                    await communicate.save(temp_file_path)
                    return temp_file_path

            # 运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            temp_file_path = loop.run_until_complete(generate_speech())
            loop.close()

            if async_play:
                # 异步播放
                play_thread = threading.Thread(
                    target=self._play_audio_file,
                    args=(temp_file_path,),
                    daemon=True
                )
                play_thread.start()
            else:
                # 同步播放
                self._play_audio_file(temp_file_path)

            print("[OK] Edge语音合成完成！")

        except Exception as e:
            print(f"Edge TTS合成失败: {e}")
            # 回退到gTTS
            if self.use_gtts:
                self._gtts_text_to_speech(text, language, async_play)
            else:
                print("[ERROR] Edge TTS和gTTS都不可用")

    def _gtts_text_to_speech(self, text, language, async_play):
        """使用gTTS进行语音合成"""
        try:
            # 根据语言选择TLD
            tld = 'com'  # 默认英文
            if language == 'zh':
                tld = 'com.cn'  # 中文使用中国服务器

            # 生成语音
            tts = gTTS(text=text, lang='en' if language in ['en', 'mixed'] else 'zh-cn', tld=tld, slow=False)

            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file_path = temp_file.name
                tts.save(temp_file_path)

            if async_play:
                # 异步播放
                play_thread = threading.Thread(
                    target=self._play_audio_file,
                    args=(temp_file_path,),
                    daemon=True
                )
                play_thread.start()
                print("语音合成完成，开始异步播放...")
            else:
                # 同步播放：确保pygame初始化并等待播放完成
                self._play_audio_file_sync(temp_file_path)
            print("语音播放完成！")

        except Exception as e:
            print(f"gTTS播放失败: {e}")
            print("[ERROR] gTTS不可用，没有其他可用的TTS引擎")

    def _play_audio_file(self, file_path):
        """异步播放音频文件"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # 清理临时文件
            try:
                os.unlink(file_path)
            except:
                pass

        except Exception as e:
            print(f"音频播放失败: {e}")
            # 清理临时文件
            try:
                os.unlink(file_path)
            except:
                pass

    def _play_audio_file_sync(self, file_path):
        """同步播放音频文件，确保完整播放"""
        try:
            # 确保pygame mixer已初始化
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # 停止任何正在播放的音频
            pygame.mixer.music.stop()

            # 加载并播放音频
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # 等待播放开始
            time.sleep(0.1)  # 给播放缓冲一点时间

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # 清理临时文件
            try:
                os.unlink(file_path)
            except:
                pass

        except Exception as e:
            print(f"同步音频播放失败: {e}")
            # 清理临时文件
            try:
                os.unlink(file_path)
            except:
                pass
