# -*- coding: utf-8 -*-
"""自动样本筛选：规则过滤 -> 模型置信度过滤 -> 输出高质量样本集
用法: python auto_filter_samples.py --src collected/ --dst collected/clean/ --conf 0.7
"""
import os
import sys
import shutil
import argparse
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
V5_ONNX = os.path.join(BASE, '..', 'hb_yolo', 'WarThunder_Yellow_Mark_Rangefinder-main',
                       'distance', 'code', 'yolo5', 'best.onnx')
if not os.path.exists(V5_ONNX):
    V5_ONNX = os.path.join(BASE, 'wt_best.onnx')

VALID_CLASSES = {0, 1, 2}


def _read_label(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((cid, cx, cy, bw, bh))
    return boxes


def rule_filter(img_path, txt_path):
    """规则过滤，返回 (通过, 原因)"""
    # 图片尺寸异常
    img = cv2.imread(img_path)
    if img is None:
        return False, '图片无法读取'
    h, w = img.shape[:2]
    if w < 32 or h < 32 or w > 4096 or h > 4096:
        return False, '图片尺寸异常'
    # 标注缺失
    boxes = _read_label(txt_path)
    if boxes is None:
        return False, '无标注文件'
    if len(boxes) == 0:
        return False, '标注为空'
    for (cid, cx, cy, bw, bh) in boxes:
        # 类别缺失/非法
        if cid not in VALID_CLASSES:
            return False, '类别非法:%d' % cid
        # 面积过小/过大
        area = bw * bh
        if area < 0.0001 or area > 0.98:
            return False, '面积异常'
        # 超出边界
        if cx - bw / 2 < 0 or cy - bh / 2 < 0 or cx + bw / 2 > 1 or cy + bh / 2 > 1:
            return False, '越界'
    return True, ''


def _parse_out(out):
    if out.ndim == 3:
        is_v8 = out.shape[1] < out.shape[2]
        return (out[0].T if is_v8 else out[0]), is_v8
    is_v8 = out.shape[0] < out.shape[1]
    return (out.T if is_v8 else out), is_v8


def confidence_score(net, img_path):
    """返回模型对该样本的最高置信度"""
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (480, 480), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()[0]
    pred, is_v8 = _parse_out(out)
    best = 0.0
    for row in pred:
        if is_v8:
            sc = float(np.max(row[4:]))
        else:
            sc = float(row[4] * np.max(row[5:]))
        if sc > best:
            best = sc
    return best


def confidence_filter(net, img_path, conf):
    """模型置信度过滤：最高置信度>conf 才保留"""
    return confidence_score(net, img_path) > conf


def main():
    p = argparse.ArgumentParser(description='自动样本筛选（规则+置信度）')
    p.add_argument('--src', default='collected', help='原始样本目录')
    p.add_argument('--dst', default=None, help='输出目录(默认 src/clean)')
    p.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    a = p.parse_args()

    src = a.src
    dst = a.dst or os.path.join(src, 'clean')
    os.makedirs(os.path.join(dst), exist_ok=True)

    net = cv2.dnn.readNetFromONNX(V5_ONNX) if os.path.exists(V5_ONNX) else None

    rule_passed = []   # (fname, conf)
    dropped = 0
    stats = {}
    total = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith('.jpg'):
            continue
        total += 1
        img_path = os.path.join(src, f)
        txt_path = os.path.join(src, os.path.splitext(f)[0] + '.txt')
        ok, reason = rule_filter(img_path, txt_path)
        if not ok:
            dropped += 1
            stats[reason or '未知'] = stats.get(reason or '未知', 0) + 1
            continue
        conf = confidence_score(net, img_path) if net is not None else 1.0
        rule_passed.append((f, conf))

    # 置信度筛选；若一个都筛不出，兜底保留规则通过的样本
    high = [x for x in rule_passed if x[1] > a.conf]
    fallback = False
    if high:
        chosen = high
    else:
        chosen = sorted(rule_passed, key=lambda x: -x[1])
        fallback = bool(chosen)

    for f, _ in chosen:
        img_path = os.path.join(src, f)
        txt_path = os.path.join(src, os.path.splitext(f)[0] + '.txt')
        shutil.copy(img_path, os.path.join(dst, f))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(dst, os.path.splitext(f)[0] + '.txt'))

    kept = len(chosen)
    dropped += len(rule_passed) - kept
    print(json.dumps({'status': 'done', 'kept': kept, 'dropped': dropped,
                      'fallback': fallback, 'reasons': stats},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
