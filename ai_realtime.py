#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时 AI 图像识别测距 - YOLOv4-tiny
持续检测屏幕上指定的物体A和物体B，实时输出距离
输出格式：每行一个 JSON 对象（流式输出）
"""

import sys
import json
import cv2
import numpy as np
import mss
import os
import time
import urllib.request

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CFG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"


def download_model():
    if not os.path.exists(WEIGHTS_PATH):
        print(json.dumps({"status": "downloading", "message": "Downloading yolov4-tiny.weights (~23MB)..."}), flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    if not os.path.exists(CFG_PATH):
        urllib.request.urlretrieve(CFG_URL, CFG_PATH)
    if not os.path.exists(NAMES_PATH):
        urllib.request.urlretrieve(NAMES_URL, NAMES_PATH)


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = np.array(shot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def load_yolo():
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DnnBackend_DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DnnTarget_DNN_TARGET_CPU)
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(net, classes, output_layers, screen, conf_threshold=0.3, nms_threshold=0.4):
    height, width = screen.shape[:2]
    blob = cv2.dnn.blobFromImage(screen, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                boxes.append([cx - w//2, cy - h//2, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            left, top, w, h = boxes[i]
            results.append({
                "label": classes[class_ids[i]],
                "confidence": round(confidences[i], 3),
                "x": round(left + w/2, 1),
                "y": round(top + h/2, 1),
                "left": int(left), "top": int(top),
                "width": int(w), "height": int(h)
            })
    return results


def find_best(detections, target_label):
    """在检测结果中找到指定标签置信度最高的目标"""
    matches = [d for d in detections if d["label"] == target_label]
    if not matches:
        return None
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    return matches[0]


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Usage: python ai_realtime.py <labelA> <labelB> [coefA] [coefB] [unit] [interval]"}), flush=True)
        sys.exit(1)

    label_a = sys.argv[1].lower().strip()
    label_b = sys.argv[2].lower().strip()
    coef_a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    coef_b = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    unit = sys.argv[5] if len(sys.argv) > 5 else "px"
    interval = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    # 下载模型
    if not os.path.exists(WEIGHTS_PATH):
        download_model()

    net, classes, output_layers = load_yolo()

    # 检查标签是否有效
    if label_a not in classes:
        print(json.dumps({"error": "Unknown label A: " + label_a + ". Available: " + ", ".join(classes[:20]) + "..."}), flush=True)
        sys.exit(1)
    if label_b not in classes:
        print(json.dumps({"error": "Unknown label B: " + label_b + ". Available: " + ", ".join(classes[:20]) + "..."}), flush=True)
        sys.exit(1)

    # 发送就绪信号
    print(json.dumps({"status": "ready", "labelA": label_a, "labelB": label_b, "model": "YOLOv4-tiny", "classes": classes}), flush=True)

    # 实时检测循环
    while True:
        try:
            screen = capture_screen()
            detections = detect_objects(net, classes, output_layers, screen)

            obj_a = find_best(detections, label_a)
            obj_b = find_best(detections, label_b)

            if obj_a and obj_b:
                dx = abs(obj_b["x"] - obj_a["x"])
                dy = abs(obj_b["y"] - obj_a["y"])
                pixel_dist = (dx**2 + dy**2) ** 0.5
                result = pixel_dist * coef_a / coef_b if coef_b != 0 else 0

                output = {
                    "status": "measured",
                    "objA": obj_a,
                    "objB": obj_b,
                    "dx": round(dx, 1),
                    "dy": round(dy, 1),
                    "pixel_distance": round(pixel_dist, 1),
                    "result": round(result, 3),
                    "unit": unit,
                    "total_detected": len(detections)
                }
            elif obj_a and not obj_b:
                output = {"status": "partial", "found": "A", "objA": obj_a, "message": "B (" + label_b + ") not found"}
            elif obj_b and not obj_a:
                output = {"status": "partial", "found": "B", "objB": obj_b, "message": "A (" + label_a + ") not found"}
            else:
                output = {"status": "searching", "message": "Looking for " + label_a + " and " + label_b + "..."}

            print(json.dumps(output, ensure_ascii=False), flush=True)

        except KeyboardInterrupt:
            print(json.dumps({"status": "stopped"}), flush=True)
            break
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

        time.sleep(interval)


if __name__ == "__main__":
    main()
