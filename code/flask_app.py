#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask后端API服务
提供文本对话、语音识别、语音合成等接口
"""

import os
import sys
import base64
import tempfile
import time
import traceback
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import soundfile as sf

# 设置控制台编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 导入现有模块
from speech_system import CompleteSpeechSystem
from audio_processor import AudioProcessor
from conversation_engine import ConversationEngine
from dialogue_policy import DialoguePolicy
from tts_engine import TTSEngine
from language_utils import LanguageUtils

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持跨域请求

# 全局变量：存储系统实例
speech_system = None
audio_processor = None
conversation_engine = None
tts_engine = None

# 创建临时文件目录
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_audio')
os.makedirs(TEMP_DIR, exist_ok=True)

HTTP_CHAT_HISTORY_LIMIT = int(os.environ.get("HTTP_CHAT_HISTORY_LIMIT", "40"))
HTTP_CHAT_HISTORIES = {}
HTTP_CHAT_HISTORY_LOCK = threading.Lock()


def init_system():
    """初始化语音对话系统"""
    global speech_system, audio_processor, conversation_engine, tts_engine

    if audio_processor is not None and conversation_engine is not None and tts_engine is not None:
        print("Flask backend components already initialized, skipping duplicate startup")
        return True
    
    print("=" * 60)
    print("正在初始化Flask后端服务...")
    print("=" * 60)
    
    try:
        # 初始化各个组件
        audio_processor = AudioProcessor()
        audio_processor.init_sensevoice()
        
        # 设置本地模型路径
        local_model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 
            'Qwen2.5-1.5B-Instruct'
        )
        
        conversation_engine = ConversationEngine()
        # 使用本地模型路径初始化
        conversation_engine.init_model(local_model_path=local_model_path)
        
        # 在服务器环境下优先使用本地TTS模型
        import sys
        server_mode = sys.platform.startswith('linux')  # 在Linux服务器上启用服务器模式
        try:
            # 使用Edge TTS（支持中英文，稳定可靠）
            tts_engine = TTSEngine(prefer_edge_tts=True, prefer_local_tts=True)
        except TypeError:
            # 如果参数不支持，回退到默认设置
            print("⚠️ TTS引擎参数不支持，使用默认配置")
            tts_engine = TTSEngine()
        tts_engine.init_tts()
        
        print("✅ 所有组件初始化成功！")
        return True
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        traceback.print_exc()
        return False


def _get_http_chat_session_id(data):
    """Pick a stable chat session id; fallback keeps local demos from forgetting."""
    return (
        data.get("session_id")
        or data.get("conversation_id")
        or data.get("client_id")
        or request.headers.get("X-Session-Id")
        or "default"
    )


def _load_http_chat_history(session_id, incoming_history):
    normalized_incoming = DialoguePolicy.normalize_history(incoming_history or [])
    with HTTP_CHAT_HISTORY_LOCK:
        if normalized_incoming:
            HTTP_CHAT_HISTORIES[session_id] = normalized_incoming[-HTTP_CHAT_HISTORY_LIMIT:]
            return list(HTTP_CHAT_HISTORIES[session_id])
        return list(HTTP_CHAT_HISTORIES.get(session_id, []))


def _save_http_chat_turn(session_id, base_history, user_message, assistant_message):
    updated_history = DialoguePolicy.normalize_history(base_history or [])
    if not updated_history or updated_history[-1] != {"role": "user", "content": user_message}:
        updated_history.append({"role": "user", "content": user_message})
    if assistant_message:
        updated_history.append({"role": "assistant", "content": assistant_message})

    with HTTP_CHAT_HISTORY_LOCK:
        HTTP_CHAT_HISTORIES[session_id] = updated_history[-HTTP_CHAT_HISTORY_LIMIT:]
        return len(HTTP_CHAT_HISTORIES[session_id])


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'success': True,
        'message': 'EchoSage API服务运行正常',
        'timestamp': time.time()
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    文本对话接口
    接收用户消息，返回AI回复
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # 验证必填参数
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '缺少必填参数: message'
                }
            }), 400
        
        user_message = data.get('message', '').strip()
        session_id = _get_http_chat_session_id(data)
        if data.get("reset_history"):
            with HTTP_CHAT_HISTORY_LOCK:
                HTTP_CHAT_HISTORIES.pop(session_id, None)
        history = _load_http_chat_history(session_id, data.get('history', []))
        mode = data.get('mode', 'normal')  # normal 或 phone
        need_audio = data.get('need_audio', False)  # 是否需要语音合成
        language_preference = data.get('language_preference', 'english')
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '消息内容不能为空'
                }
            }), 400
        
        # 构建对话上下文。即使前端漏传 history，也用服务端会话历史兜底。
        context_prompt = DialoguePolicy.build_chat_prompt(
            history,
            user_message,
            max_history_messages=20,
        )
        print(
            f"Chat session={session_id}, incoming_history={len(data.get('history', []) or [])}, "
            f"effective_history={len(history)}, intent={DialoguePolicy.classify_user_intent(user_message, history)}"
        )
        response_text = conversation_engine.generate_response(
            context_prompt,
            use_context=False,
            allow_long_response=(mode == 'normal'),
            medium_response=False
        )
        history_count = _save_http_chat_turn(session_id, history, user_message, response_text)
        
        # 检测语言
        detected_language = LanguageUtils.detect_text_language(response_text)

        # 根据need_audio参数决定是否生成语音
        audio_url = None
        duration = 0
        tts_time = 0

        if need_audio:
            # 生成语音
            tts_start_time = time.time()
            try:
                # 使用TTS引擎生成语音文件（不播放）
                audio_file_path = tts_engine.generate_speech_file(
                    response_text,
                    save_dir=TEMP_DIR
                )

                # 生成音频URL
                if audio_file_path and os.path.exists(audio_file_path):
                    file_id = os.path.basename(audio_file_path)
                    audio_url = f'/api/audio/{file_id}'

                    # 计算音频时长
                    try:
                        audio_data = audio_processor.load_audio_from_file(audio_file_path)
                        duration = len(audio_data) / audio_processor.sample_rate if audio_data is not None else 0
                    except:
                        duration = 0
                else:
                    print("⚠️ TTS生成语音失败，仅返回文字")

            except Exception as e:
                print(f"⚠️ TTS生成失败: {e}")
                traceback.print_exc()

            tts_time = (time.time() - tts_start_time) * 1000

        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 构建响应数据
        response_data = {
            'text': response_text,
            'metadata': {
                'tokens_used': len(response_text),  # 简化的token计数
                'processing_time': round(processing_time, 2),
                'language_detected': detected_language,
                'session_id': session_id,
                'history_count': history_count
            }
        }

        # 如果需要音频，添加音频相关字段
        if need_audio:
            response_data['audio_url'] = audio_url
            response_data['metadata']['duration'] = round(duration, 2)
            response_data['metadata']['tts_time'] = round(tts_time, 2)
        else:
            response_data['audio_url'] = None

        return jsonify({
            'success': True,
            'data': response_data,
            'message': '处理成功'
        })
        
    except Exception as e:
        print(f"❌ 对话处理失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_FAILED',
                'message': f'对话处理失败: {str(e)}'
            }
        }), 500


@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    语音转文字接口
    接收base64编码的音频数据，返回识别的文字
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        print(f"🔍 收到请求数据: {data}")

        # 支持两种方式：base64音频数据 或 file_id
        audio_base64 = data.get('audio', '')
        file_id = data.get('file_id', '')
        audio_format = data.get('format', 'wav')
        language = data.get('language', 'auto')

        print(f"📊 参数: audio长度={len(audio_base64)}, file_id={file_id}, format={audio_format}")

        temp_file_path = None
        should_delete = False

        # 方式1: 使用file_id（前端已上传文件）
        if file_id:
            temp_file_path = os.path.join(TEMP_DIR, file_id)
            print(f"📂 尝试读取文件: {temp_file_path}")
            print(f"📁 TEMP_DIR: {TEMP_DIR}")
            print(f"📄 file_id: {file_id}")
            print(f"✓ 文件存在: {os.path.exists(temp_file_path)}")

            if not os.path.exists(temp_file_path):
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'FILE_NOT_FOUND',
                        'message': f'文件不存在: {file_id}'
                    }
                }), 400
            should_delete = False  # 不删除，可能还需要使用
            print(f"✅ 使用file_id方式: {file_id}")

        # 方式2: 使用base64音频数据
        elif audio_base64:
            print(f"✅ 使用base64方式, 数据长度: {len(audio_base64)}")
            # 解码base64音频数据
            try:
                audio_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'INVALID_AUDIO_FORMAT',
                        'message': f'音频数据解码失败: {str(e)}'
                    }
                }), 400

            # 保存到临时文件
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f'.{audio_format}',
                dir=TEMP_DIR
            )
            temp_file.write(audio_bytes)
            temp_file.close()
            temp_file_path = temp_file.name
            should_delete = True

        else:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '缺少必填参数: audio 或 file_id'
                }
            }), 400

        try:
            # 加载音频文件
            audio_data = audio_processor.load_audio_from_file(temp_file_path)
            
            if audio_data is None:
                return jsonify({
                    'success': False,
                    'error': {
                        'code': 'INVALID_AUDIO_FORMAT',
                        'message': '不支持的音频格式或文件损坏'
                    }
                }), 400
            
            # 计算音频时长
            duration = len(audio_data) / audio_processor.sample_rate
            
            # 语音识别
            recognized_text = audio_processor.speech_to_text(audio_data)
            
            # 检测语言
            detected_language = audio_processor._detect_text_language(recognized_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'success': True,
                'data': {
                    'text': recognized_text,
                    'language': detected_language,
                    'confidence': 0.95,  # SenseVoice 当前未返回统一置信度，先使用固定值
                    'duration': round(duration, 2),
                    'metadata': {
                        'processing_time': round(processing_time, 2),
                        'model_used': audio_processor.asr_backend_name
                    }
                },
                'message': '语音识别成功'
            })
            
        finally:
            # 清理临时文件（仅当是base64上传的临时文件时才删除）
            if should_delete and temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
    except Exception as e:
        print(f"❌ 语音识别失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_FAILED',
                'message': f'语音识别失败: {str(e)}'
            }
        }), 500


@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """
    文字转语音接口
    接收文本，返回生成的音频文件URL
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # 验证必填参数
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '缺少必填参数: text'
                }
            }), 400
        
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')
        voice = data.get('voice', 'default')
        speed = data.get('speed', 1.0)
        output_format = data.get('format', 'mp3')
        
        if not text:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '文本内容不能为空'
                }
            }), 400
        
        # 检测语言
        if language == 'auto':
            language = LanguageUtils.detect_text_language(text)

        audio_file_path = tts_engine.generate_speech_file(text, save_dir=TEMP_DIR)
        if not audio_file_path or not os.path.exists(audio_file_path):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'TTS_FAILED',
                    'message': 'TTS生成失败：没有可用的语音合成引擎'
                }
            }), 500

        file_size = os.path.getsize(audio_file_path)
        estimated_duration = len(text) * 0.1
        processing_time = (time.time() - start_time) * 1000
        file_id = os.path.basename(audio_file_path)
        audio_url = f'/api/audio/{file_id}'

        return jsonify({
            'success': True,
            'data': {
                'audio_url': audio_url,
                'duration': round(estimated_duration, 2),
                'size': file_size,
                'metadata': {
                    'processing_time': round(processing_time, 2),
                    'voice_used': language,
                    'engine': tts_engine.get_current_engine_info()
                }
            },
            'message': 'TTS生成成功'
        })
        
        # 生成语音文件
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp3',
            dir=TEMP_DIR
        )
        temp_file.close()
        
        # 使用TTS引擎生成语音（同步方式）
        import asyncio
        import edge_tts
        
        async def generate_speech():
            # 选择合适的语音
            if language == 'zh':
                voice_name = "zh-CN-XiaoxiaoNeural"
            elif language == 'en':
                voice_name = "en-US-AriaNeural"
            else:
                voice_name = "en-US-AriaNeural"
            
            communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(temp_file.name)
        
        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech())
        loop.close()
        
        # 获取文件大小和时长
        file_size = os.path.getsize(temp_file.name)
        
        # 估算音频时长（简化计算）
        estimated_duration = len(text) * 0.1  # 粗略估算
        
        processing_time = (time.time() - start_time) * 1000
        
        # 生成文件URL（使用文件名）
        file_id = os.path.basename(temp_file.name)
        audio_url = f'/api/audio/{file_id}'
        
        return jsonify({
            'success': True,
            'data': {
                'audio_url': audio_url,
                'duration': round(estimated_duration, 2),
                'size': file_size,
                'metadata': {
                    'processing_time': round(processing_time, 2),
                    'voice_used': language
                }
            },
            'message': '语音合成成功'
        })
        
    except Exception as e:
        print(f"❌ 语音合成失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': {
                'code': 'TTS_FAILED',
                'message': f'语音合成失败: {str(e)}'
            }
        }), 500


@app.route('/api/audio/<file_id>', methods=['GET'])
def get_audio(file_id):
    """
    获取音频文件
    """
    try:
        file_path = os.path.join(TEMP_DIR, file_id)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'FILE_NOT_FOUND',
                    'message': '音频文件不存在'
                }
            }), 404
        
        ext = os.path.splitext(file_path)[1].lower()
        mimetype = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
        }.get(ext, "application/octet-stream")

        return send_file(
            file_path,
            mimetype=mimetype,
            as_attachment=False
        )
        
    except Exception as e:
        print(f"❌ 获取音频文件失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'FILE_ACCESS_FAILED',
                'message': f'获取音频文件失败: {str(e)}'
            }
        }), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    文件上传接口
    接收音频文件，返回文件信息
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '缺少文件参数'
                }
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PARAMS',
                    'message': '文件名为空'
                }
            }), 400
        
        # 保存文件
        file_ext = os.path.splitext(file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=TEMP_DIR
        )
        file.save(temp_file.name)
        temp_file.close()

        print(f"📁 文件上传成功: {temp_file.name}")
        print(f"📊 文件大小: {os.path.getsize(temp_file.name)} bytes")

        # 获取文件信息
        file_size = os.path.getsize(temp_file.name)
        file_id = os.path.basename(temp_file.name)
        file_url = f'/api/audio/{file_id}'
        recognized_text = ""
        detected_language = "en"

        # 尝试验证音频文件
        try:
            print(f"🔍 验证音频文件格式...")
            audio_data = audio_processor.load_audio_from_file(temp_file.name)
            if audio_data is not None:
                duration = len(audio_data) / audio_processor.sample_rate
                recognized_text = audio_processor.speech_to_text(audio_data)
                detected_language = audio_processor._detect_text_language(recognized_text)
                print(f"Upload ASR result: '{recognized_text}'")
                print(f"✅ 音频文件验证通过，时长: {duration:.2f}秒")
            else:
                duration = 0
                print(f"⚠️ 音频文件验证失败，但仍然保存")
        except Exception as e:
            duration = 0
            print(f"⚠️ 音频文件验证失败: {e}")
            import traceback
            traceback.print_exc()
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_url': file_url,
            'text': recognized_text,
            'data': {
                'file_id': file_id,
                'file_url': file_url,
                'text': recognized_text,
                'language': detected_language,
                'duration': round(duration, 2),
                'size': file_size,
                'metadata': {
                    'model_used': audio_processor.asr_backend_name
                }
            },
            'message': '文件上传成功'
        })
        
    except Exception as e:
        print(f"❌ 文件上传失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': {
                'code': 'UPLOAD_FAILED',
                'message': f'文件上传失败: {str(e)}'
            }
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': '请求的资源不存在'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        'success': False,
        'error': {
            'code': 'INTERNAL_ERROR',
            'message': '服务器内部错误'
        }
    }), 500


if __name__ == '__main__':
    # 初始化系统
    if init_system():
        print("=" * 60)
        print("🚀 Flask后端服务启动成功！")
        print("📡 API地址: http://localhost:8000")
        print("=" * 60)
        
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=8000,
            debug=False,  # 生产环境关闭debug
            threaded=True  # 启用多线程支持
        )
    else:
        print("❌ 系统初始化失败，无法启动服务")
        sys.exit(1)
