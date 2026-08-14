#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 图像识别测距 - 使用 OpenCV DNN + YOLOv4-tiny
自动检测屏幕上的物体，返回位置坐标用于测距
"""

import sys
import json
import cv2
import numpy as np
import mss
import os
import urllib.request

# 模型文件路径（与脚本同目录）
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
CFG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

# 下载URL
WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"


def download_model():
    """下载 YOLO 模型文件"""
    if not os.path.exists(WEIGHTS_PATH):
        print(json.dumps({"status": "downloading_weights", "message": "Downloading yolov4-tiny.weights (~23MB)..."}))
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    if not os.path.exists(CFG_PATH):
        urllib.request.urlretrieve(CFG_URL, CFG_PATH)
    if not os.path.exists(NAMES_PATH):
        urllib.request.urlretrieve(NAMES_URL, NAMES_PATH)


def capture_screen():
    """截取主屏幕"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = np.array(shot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def load_yolo():
    """加载 YOLO 模型"""
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DnnBackend_DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DnnTarget_DNN_TARGET_CPU)
    
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 获取输出层名
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers


def detect_objects(net, classes, output_layers, screen, conf_threshold=0.3, nms_threshold=0.4):
    """运行 YOLO 目标检测"""
    height, width = screen.shape[:2]
    
    # 预处理：缩放到416x416，归一化
    blob = cv2.dnn.blobFromImage(screen, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 前向推理
    outputs = net.forward(output_layers)
    
    # 解析结果
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # 转换回原图坐标
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                left = cx - w // 2
                top = cy - h // 2
                
                boxes.append([left, top, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # 非极大值抑制（NMS）去重
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, w, h = box
            results.append({
                "label": classes[class_ids[i]],
                "confidence": round(confidences[i], 3),
                "x": round(left + w / 2, 1),   # 中心X
                "y": round(top + h / 2, 1),     # 中心Y
                "left": int(left),
                "top": int(top),
                "width": int(w),
                "height": int(h)
            })
    
    return results


def main():
    # 下载模型（首次运行）
    if not os.path.exists(WEIGHTS_PATH):
        download_model()
    
    # 检查参数
    coef_a = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    coef_b = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    unit = sys.argv[3] if len(sys.argv) > 3 else "px"
    
    # 加载模型
    net, classes, output_layers = load_yolo()
    
    # 截取屏幕
    screen = capture_screen()
    
    # AI 检测
    detections = detect_objects(net, classes, output_layers, screen)
    
    if len(detections) < 2:
        print(json.dumps({"error": "Not enough objects detected (found " + str(len(detections)) + ", need >= 2)", "detections": detections}))
        sys.exit(0)
    
    # 取置信度最高的两个目标
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    obj1 = detections[0]
    obj2 = detections[1]
    
    # 计算距离
    dx = abs(obj2["x"] - obj1["x"])
    dy = abs(obj2["y"] - obj1["y"])
    pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
    result = pixel_dist * coef_a / coef_b if coef_b != 0 else 0
    
    output = {
        "obj1": obj1,
        "obj2": obj2,
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "pixel_distance": round(pixel_dist, 1),
        "coefA": coef_a,
        "coefB": coef_b,
        "result": round(result, 3),
        "unit": unit,
        "total_detected": len(detections),
        "all_detections": detections[:10],  # 返回前10个检测结果
        "model": "YOLOv4-tiny",
        "classes": classes
    }
    
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
