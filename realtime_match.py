#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时双图模板匹配测距
持续截屏 -> 用 OpenCV 模板匹配定位两张参考图 -> 输出 JSON 行
被 Electron 主进程 spawn，通过 stdout 流式输出
"""

import sys
import json
import cv2
import numpy as np
import mss
import time


def find_template(screen, template):
    """在屏幕截图中查找模板位置，返回中心坐标和匹配度"""
    if template is None or screen is None:
        return None
    th, tw = template.shape[:2]
    sh, sw = screen.shape[:2]
    if tw > sw or th > sh:
        return None
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    cx = max_loc[0] + tw // 2
    cy = max_loc[1] + th // 2
    return {"x": int(cx), "y": int(cy), "score": round(float(max_val), 4), "w": int(tw), "h": int(th)}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: realtime_match.py <imgA> <imgB> [interval]"}), flush=True)
        sys.exit(1)

    img_a_path = sys.argv[1]
    img_b_path = sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    # 加载参考图
    img_a = cv2.imread(img_a_path, cv2.IMREAD_COLOR)
    img_b = cv2.imread(img_b_path, cv2.IMREAD_COLOR)

    if img_a is None:
        print(json.dumps({"error": "Cannot load image A: " + img_a_path}), flush=True)
        sys.exit(1)
    if img_b is None:
        print(json.dumps({"error": "Cannot load image B: " + img_b_path}), flush=True)
        sys.exit(1)

    print(json.dumps({"status": "ready", "imgA": img_a_path, "imgB": img_b_path}), flush=True)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            try:
                shot = sct.grab(monitor)
                screen = np.array(shot)
                screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

                pos_a = find_template(screen, img_a)
                pos_b = find_template(screen, img_b)

                if pos_a and pos_b:
                    dx = abs(pos_b["x"] - pos_a["x"])
                    dy = abs(pos_b["y"] - pos_a["y"])
                    pixel_dist = round((dx ** 2 + dy ** 2) ** 0.5, 1)
                    output = {
                        "status": "measured",
                        "a": pos_a,
                        "b": pos_b,
                        "dx": int(dx),
                        "dy": int(dy),
                        "pixel_distance": pixel_dist
                    }
                elif pos_a:
                    output = {"status": "partial", "found": "A", "a": pos_a}
                elif pos_b:
                    output = {"status": "partial", "found": "B", "b": pos_b}
                else:
                    output = {"status": "searching"}

                print(json.dumps(output), flush=True)

            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

            time.sleep(interval)


if __name__ == "__main__":
    main()
