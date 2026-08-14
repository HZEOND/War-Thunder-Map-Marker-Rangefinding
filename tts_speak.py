#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
曼波口音语音播报 - 基于 edge-tts (免费, 无需API Key)
输入: 文本 + 音色/语速/音调参数
输出: JSON 包含 base64 编码的 MP3 音频
被 Electron 主进程 spawn 调用
"""

import sys
import json
import asyncio
import base64
import os
import tempfile

import edge_tts


async def generate_speech(text, voice, rate, pitch):
    """生成语音并返回 base64"""
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )
    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    tmp_path = tmp.name
    tmp.close()

    await communicate.save(tmp_path)

    with open(tmp_path, 'rb') as f:
        audio_data = f.read()

    # 清理临时文件
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return base64.b64encode(audio_data).decode('utf-8')


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: tts_speak.py <text> [voice] [rate] [pitch]"}))
        sys.exit(1)

    text = sys.argv[1]
    # 默认曼波口音配置: 晓伊(年轻可爱女声) + 稍快语速 + 升调
    voice = sys.argv[2] if len(sys.argv) > 2 else "zh-CN-XiaoyiNeural"
    rate = sys.argv[3] if len(sys.argv) > 3 else "+10%"
    pitch = sys.argv[4] if len(sys.argv) > 4 else "+8Hz"

    try:
        audio_b64 = asyncio.run(generate_speech(text, voice, rate, pitch))
        print(json.dumps({
            "audio": audio_b64,
            "format": "mp3",
            "voice": voice
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
