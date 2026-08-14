#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视觉测距 - OpenCV 屏幕图标识别与距离计算
功能：截取屏幕 -> 用 OpenCV 模板匹配定位两个图标 -> 计算欧几里得距离
公式：d = SQRT(x^2 + y^2) * (a / b)
"""

import sys
import json
import cv2
import numpy as np
import mss


def capture_screen():
    """截取主屏幕，返回 BGR numpy 数组"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = np.array(shot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def find_icon(screen, icon, threshold=0.5):
    """
    使用 OpenCV 模板匹配在屏幕中定位图标
    算法：TM_CCOEFF_NORMED（归一化相关系数）
    """
    if icon is None or screen is None:
        return None

    ih, iw = icon.shape[:2]
    sh, sw = screen.shape[:2]
    if iw > sw or ih > sh:
        return None

    # 模板匹配
    result = cv2.matchTemplate(screen, icon, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    left, top = max_loc
    return {
        "x": round(left + iw / 2, 1),
        "y": round(top + ih / 2, 1),
        "left": int(left),
        "top": int(top),
        "width": int(iw),
        "height": int(ih),
        "score": round(float(max_val), 4)
    }


def find_all_matches(screen, icon, threshold=0.8, max_results=10):
    """查找屏幕上所有匹配位置"""
    if icon is None or screen is None:
        return []
    ih, iw = icon.shape[:2]
    sh, sw = screen.shape[:2]
    if iw > sw or ih > sh:
        return []
    result = cv2.matchTemplate(screen, icon, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= threshold)
    matches = []
    for pt in zip(*locs[::-1]):
        # 去重：如果与已有点距离太近则跳过
        too_close = False
        for m in matches:
            if abs(m["left"] - pt[0]) < iw/2 and abs(m["top"] - pt[1]) < ih/2:
                too_close = True
                break
        if not too_close:
            matches.append({
                "x": round(pt[0] + iw/2, 1),
                "y": round(pt[1] + ih/2, 1),
                "left": int(pt[0]),
                "top": int(pt[1]),
                "width": int(iw),
                "height": int(ih),
                "score": round(float(result[pt[1], pt[0]]), 4)
            })
        if len(matches) >= max_results:
            break
    return matches


def edge_detect(screen_gray, x, y, radius=30):
    """在指定位置进行边缘检测，辅助精确测量"""
    h, w = screen_gray.shape[:2]
    x, y = int(x), int(y)
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    roi = screen_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 找最大轮廓的中心
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1
    return {"x": cx, "y": cy}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python visual_measure.py <icon1> <icon2> [a] [b] [unit]"}))
        sys.exit(1)

    icon1_path = sys.argv[1]
    icon2_path = sys.argv[2]
    coef_a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    coef_b = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    unit = sys.argv[5] if len(sys.argv) > 5 else "px"

    # 加载图标
    icon1 = cv2.imread(icon1_path, cv2.IMREAD_COLOR)
    icon2 = cv2.imread(icon2_path, cv2.IMREAD_COLOR)
    if icon1 is None:
        print(json.dumps({"error": "Cannot load icon1: " + icon1_path}))
        sys.exit(1)
    if icon2 is None:
        print(json.dumps({"error": "Cannot load icon2: " + icon2_path}))
        sys.exit(1)

    # 截取屏幕
    screen = capture_screen()
    if screen is None:
        print(json.dumps({"error": "Cannot capture screen"}))
        sys.exit(1)

    # 转灰度用于边缘检测
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # OpenCV 模板匹配
    match1 = find_icon(screen, icon1)
    match2 = find_icon(screen, icon2)

    if match1 is None or match2 is None:
        print(json.dumps({"error": "Icon not found on screen"}))
        sys.exit(1)

    # 边缘检测精确定位
    refined1 = edge_detect(screen_gray, match1["x"], match1["y"])
    refined2 = edge_detect(screen_gray, match2["x"], match2["y"])
    if refined1:
        match1["refined"] = refined1
    if refined2:
        match2["refined"] = refined2

    # 计算距离
    dx = abs(match2["x"] - match1["x"])
    dy = abs(match2["y"] - match1["y"])
    pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
    result = pixel_dist * coef_a / coef_b if coef_b != 0 else 0

    output = {
        "icon1": match1,
        "icon2": match2,
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "pixel_distance": round(pixel_dist, 1),
        "coefA": coef_a,
        "coefB": coef_b,
        "result": round(result, 3),
        "unit": unit,
        "screen_width": int(screen.shape[1]),
        "screen_height": int(screen.shape[0]),
        "opencv_version": cv2.__version__
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
