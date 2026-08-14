#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
区域实时双图 AI 识别测距（增强版）
多方法匹配：灰度模板 + 边缘模板，取最高分
在指定屏幕区域内持续检测两张参考图，输出两者中心点距离
被 Electron 主进程 spawn，通过 stdout 逐行输出 JSON
"""

import sys
import json
import cv2
import numpy as np
import mss
import time


def imread_unicode(path):
    """读取含中文路径的图片"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def multi_scale_match(region_gray, region_edge, tmpl, scales, threshold):
    """
    多尺度 + 多方法模板匹配
    同时在灰度图和边缘图上匹配，返回最高分结果
    返回 dict 或 None
    """
    rh, rw = region_gray.shape[:2]
    tg = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    te = cv2.Canny(tg, 80, 180)
    th, tw = tg.shape[:2]

    best = None  # (score, cx, cy, scale, method)

    for s in scales:
        nw, nh = int(tw * s), int(th * s)
        if nw < 3 or nh < 3 or nw > rw or nh > rh:
            continue
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR

        # 灰度匹配
        try:
            sg = cv2.resize(tg, (nw, nh), interpolation=interp)
            res = cv2.matchTemplate(region_gray, sg, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if best is None or mv > best[0]:
                best = (float(mv), ml[0] + nw // 2, ml[1] + nh // 2, s, 'gray')
        except cv2.error:
            pass

        # 边缘匹配
        try:
            se = cv2.resize(te, (nw, nh), interpolation=interp)
            res = cv2.matchTemplate(region_edge, se, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if best is None or mv > best[0]:
                best = (float(mv), ml[0] + nw // 2, ml[1] + nh // 2, s, 'edge')
        except cv2.error:
            pass

    if best and best[0] >= threshold:
        return {
            "x": int(best[1]), "y": int(best[2]),
            "score": round(best[0], 4),
            "scale": round(best[3], 2),
            "method": best[4]
        }
    return None


def detect_in_region(region, img_a, img_b, threshold, scales):
    """在区域中检测两个目标"""
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region_edge = cv2.Canny(region_gray, 80, 180)

    result = {"a": None, "b": None}
    if img_a is not None:
        result["a"] = multi_scale_match(region_gray, region_edge, img_a, scales, threshold)
    if img_b is not None:
        result["b"] = multi_scale_match(region_gray, region_edge, img_b, scales, threshold)
    return result


def main():
    if len(sys.argv) < 6:
        print(json.dumps({"error": "Usage: region_realtime.py <imgA> <imgB> <x> <y> <w> <h> [threshold] [interval]"}), flush=True)
        sys.exit(1)

    img_a_path = sys.argv[1]
    img_b_path = sys.argv[2]
    rx = int(sys.argv[3])
    ry = int(sys.argv[4])
    rw = int(sys.argv[5])
    rh = int(sys.argv[6])
    threshold = float(sys.argv[7]) if len(sys.argv) > 7 else 0.6
    interval = float(sys.argv[8]) if len(sys.argv) > 8 else 0.8

    img_a = imread_unicode(img_a_path)
    img_b = imread_unicode(img_b_path)

    if img_a is None:
        print(json.dumps({"error": "Cannot load image A: " + img_a_path}), flush=True)
        sys.exit(1)
    if img_b is None:
        print(json.dumps({"error": "Cannot load image B: " + img_b_path}), flush=True)
        sys.exit(1)

    if rw <= 0 or rh <= 0:
        print(json.dumps({"error": "Invalid region size"}), flush=True)
        sys.exit(1)

    # 多尺度列表（覆盖游戏内可能的缩放）
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]

    print(json.dumps({
        "status": "ready",
        "region": {"x": rx, "y": ry, "w": rw, "h": rh},
        "imgA": {"w": int(img_a.shape[1]), "h": int(img_a.shape[0])},
        "imgB": {"w": int(img_b.shape[1]), "h": int(img_b.shape[0])},
        "threshold": threshold,
        "methods": ["gray", "edge"]
    }), flush=True)

    with mss.mss() as sct:
        monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
        while True:
            try:
                shot = sct.grab(monitor)
                region = np.array(shot)
                region = cv2.cvtColor(region, cv2.COLOR_BGRA2BGR)

                det = detect_in_region(region, img_a, img_b, threshold, scales)

                if det["a"] and det["b"]:
                    dx = det["b"]["x"] - det["a"]["x"]
                    dy = det["b"]["y"] - det["a"]["y"]
                    pixel_dist = round((dx ** 2 + dy ** 2) ** 0.5, 1)
                    abs_a = {"x": det["a"]["x"] + rx, "y": det["a"]["y"] + ry}
                    abs_b = {"x": det["b"]["x"] + rx, "y": det["b"]["y"] + ry}
                    output = {
                        "status": "measured",
                        "a": {**det["a"], "abs": abs_a},
                        "b": {**det["b"], "abs": abs_b},
                        "dx": int(abs(dx)),
                        "dy": int(abs(dy)),
                        "pixel_distance": pixel_dist
                    }
                elif det["a"]:
                    output = {"status": "partial", "found": "A", "a": det["a"]}
                elif det["b"]:
                    output = {"status": "partial", "found": "B", "b": det["b"]}
                else:
                    output = {"status": "searching"}

                print(json.dumps(output), flush=True)

            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

            time.sleep(interval)


if __name__ == "__main__":
    main()
