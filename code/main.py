#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别端到端系统主入口
"""

import os
import sys

def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='完整的语音对话系统')
    parser.add_argument('--file', '-f', type=str, help='指定音频文件路径进行处理')
    parser.add_argument('--list', '-l', action='store_true', help='列出当前目录的所有wav文件')

    args = parser.parse_args()

    # 动态导入以避免循环依赖
    from speech_system import CompleteSpeechSystem

    # 设置默认本地模型路径
    local_model_path = None

    # 获取项目根目录（main.py在code目录中，所以需要向上查找）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 优先检查Qwen2.5-0.5B-Instruct
    qwen_path = os.path.join(project_root, "models", "Qwen2.5-0.5B-Instruct")
    if os.path.exists(qwen_path):
        local_model_path = qwen_path
        print(f"发现本地Qwen模型: {qwen_path}")
    else:
        # 检查DialoGPT-small作为备用
        dialogpt_path = os.path.join(project_root, "models", "DialoGPT-small")
        if os.path.exists(dialogpt_path):
            local_model_path = dialogpt_path
            print(f"发现本地DialoGPT模型: {dialogpt_path}")
        else:
            print("未找到本地模型，将使用在线加载")

    # 创建系统实例
    try:
        system = CompleteSpeechSystem(local_model_path=local_model_path)
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return


    # 处理命令
    if args.list:
        wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
        if wav_files:
            print("当前目录的wav文件:")
            for i, f in enumerate(wav_files, 1):
                print(f"  {i}. {f}")
        else:
            print("当前目录没有找到wav文件")
        return

    # 开始对话
    if args.file:
        # 检查文件是否存在
        if not os.path.exists(args.file):
            print(f"错误：文件不存在 '{args.file}'")
            print("使用 --list 参数查看可用文件")
            return
        system.run_conversation(audio_file=args.file)
    else:
        print("用法:")
        print("  python main.py --file test1.wav          # 处理单个文件")
        print("  python main.py --list                     # 列出所有wav文件")

if __name__ == "__main__":
    main()
