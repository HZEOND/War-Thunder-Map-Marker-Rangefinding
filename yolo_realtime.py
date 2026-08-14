#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO 实时目标检测测距（使用参考项目真实训练的 YOLOv5 模型）
模型在真实战争雷霆小地图数据上训练，3 类:
  class 0/1 = 坦克箭头(jiantou), class 2 = 黄标(huangbiao)
检测每 detect_interval 秒一次，测距每 measure_interval 秒一次
被 Electron 主进程 spawn，stdout 逐行输出 JSON
"""
import sys
import json
import os
import cv2
import numpy as np
import time

# 打包后(frozen)模型放在 exe 同目录；否则放在脚本同目录
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'wt_best.onnx')
MODEL2_PATH = os.path.join(BASE_DIR, 'wt_best_v8.onnx')   # 可选第二模型(v8n)用于融合

INPUT_SIZE = 480
# class 0/1 -> 箭头, class 2 -> 黄标
CLASS_MAP = {0: 'jiantou', 1: 'jiantou', 2: 'huangbiao'}


MAX_SAMPLES = 1500   # 样本上限，超出时 FIFO 删除最早样本


class SampleCollector:
    """异步采集：识别主循环只入队(不阻塞)，后台线程写盘，采集与识别互不干扰"""
    def __init__(self, collect_dir, max_samples=MAX_SAMPLES, min_score=0.5):
        import threading, queue
        self.dir = collect_dir
        self.max = max_samples
        self.min_score = min_score
        self.count = 0
        self.paused = False   # 训练期间暂停采集（通过 stdin 控制）
        self.last_save = 0.0
        self._queue = queue.Queue(maxsize=8)
        self._stop = False
        if collect_dir:
            os.makedirs(collect_dir, exist_ok=True)
            self.count = len([f for f in os.listdir(collect_dir) if f.endswith('.jpg')])
            self._thread = threading.Thread(target=self._writer, daemon=True)
            self._thread.start()
        # stdin 控制命令监听：pause / resume（训练期间暂停采集）
        self._stdin_thread = threading.Thread(target=self._stdin_listener, daemon=True)
        self._stdin_thread.start()

    def _stdin_listener(self):
        """监听 stdin 控制命令，避免训练时与采集冲突"""
        import sys
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd == 'pause':
                    self.paused = True
                    print(json.dumps({'status': 'collect_paused'}), flush=True)
                elif cmd == 'resume':
                    self.paused = False
                    print(json.dumps({'status': 'collect_resumed'}), flush=True)
        except Exception:
            pass

    def _evict_oldest(self):
        try:
            jpgs = [os.path.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith('.jpg')]
            if not jpgs:
                return
            oldest = min(jpgs, key=os.path.getmtime)
            os.remove(oldest)
            txt = os.path.splitext(oldest)[0] + '.txt'
            if os.path.exists(txt):
                os.remove(txt)
            self.count = max(0, self.count - 1)
        except Exception:
            pass

    def _writer(self):
        """后台写盘线程"""
        while not self._stop:
            try:
                item = self._queue.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            region, good, now = item
            try:
                if self.count >= self.max:
                    self._evict_oldest()
                H, W = region.shape[:2]
                name = 'train_%d' % int(now * 1000)
                cv2.imwrite(os.path.join(self.dir, name + '.jpg'), region)
                lines = []
                for d in good:
                    cid = d['class_id']
                    lines.append(f"{cid} {d['x']/W:.6f} {d['y']/H:.6f} {d['w']/W:.6f} {d['h']/H:.6f}")
                with open(os.path.join(self.dir, name + '.txt'), 'w') as f:
                    f.write('\n'.join(lines))
                self.count += 1
            except Exception:
                pass

    def collect(self, region, detections, now):
        """非阻塞：只入队，立即返回"""
        if not self.dir or self.paused:   # 训练期间暂停采集
            return 0
        if now - self.last_save < 3.0:   # 每3秒最多提交一张
            return 0
        good = [d for d in detections if d['score'] >= self.min_score and d['class'] in ('jiantou', 'huangbiao')]
        if not good:
            return 0
        self.last_save = now
        try:
            self._queue.put_nowait((region.copy(), good, now))
            return 1
        except Exception:
            return 0

    def release(self):
        self._stop = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass


class ScreenGrabber:
    """截图底层：
    capture_mode=0：主显示器，优先 DXcam（Desktop Duplication，高性能低延迟），失败回退 mss
    capture_mode=1：副显示器/多屏，用 mss 绝对屏幕坐标（天然支持多显示器）"""
    def __init__(self, rx, ry, rw, rh, capture_mode=0):
        self.region = (rx, ry, rw, rh)
        self.mode = capture_mode
        self.backend = 'mss'
        self.cam = None
        self.sct = None
        self.monitor = None
        if capture_mode == 0:
            try:
                import dxcam
                self.cam = dxcam.create(output_color="BGR")
                self.cam.start(target_fps=60, video_mode=False)
                self.backend = 'dxcam'
            except Exception:
                self.cam = None
        if self.cam is None:
            # mss：绝对屏幕坐标（rx/ry 为虚拟屏幕物理像素），多显示器/DPI 均正确
            try:
                import mss
                self.sct = mss.mss()
                self.monitor = {'top': ry, 'left': rx, 'width': rw, 'height': rh}
                self.backend = 'mss'
            except Exception:
                self.sct = None

    def grab(self):
        rx, ry, rw, rh = self.region
        if self.backend == 'dxcam' and self.cam is not None:
            frame = self.cam.get_latest_frame()
            if frame is None:
                return None
            return frame[ry:ry + rh, rx:rx + rw]
        if self.sct is not None:
            shot = self.sct.grab(self.monitor)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        return None

    def release(self):
        try:
            if self.cam:
                self.cam.stop()
        except Exception:
            pass


def load_net():
    if not os.path.exists(MODEL_PATH):
        return None
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # 不调用 setPreferableTarget：新图引擎不支持，且默认即 CPU，避免警告
    return net


def preprocess(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    return blob, (w, h)


def postprocess(outs, orig_size, conf_thres, nms_thres):
    """兼容 YOLOv5 (N,C)/[1,N,C] 与 YOLOv8 [1,C,N] 输出"""
    orig_w, orig_h = orig_size
    out = outs[0]
    # 归一化为 (N, C)
    if out.ndim == 3:
        if out.shape[1] < out.shape[2]:   # [1, C, N] -> YOLOv8
            pred = out[0].T
            is_v8 = True
        else:                              # [1, N, C] -> YOLOv5
            pred = out[0]
            is_v8 = False
    else:                                  # 2D
        if out.shape[0] < out.shape[1]:   # [C, N] -> 转置为 (N, C)
            pred = out.T
            is_v8 = True
        else:                              # [N, C]
            pred = out
            is_v8 = False

    boxes, scores, class_ids = [], [], []
    for row in pred:
        if is_v8:
            cx, cy, bw, bh = row[0:4]
            cls_probs = row[4:]
            cid = int(np.argmax(cls_probs))
            score = float(cls_probs[cid])
        else:
            cx, cy, bw, bh, obj = row[0:5]
            cls_probs = row[5:]
            cid = int(np.argmax(cls_probs))
            score = float(obj * cls_probs[cid])
        if score < conf_thres:
            continue
        x1 = (cx - bw / 2) * orig_w / INPUT_SIZE
        y1 = (cy - bh / 2) * orig_h / INPUT_SIZE
        x2 = (cx + bw / 2) * orig_w / INPUT_SIZE
        y2 = (cy + bh / 2) * orig_h / INPUT_SIZE
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(score)
        class_ids.append(cid)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    results = []
    if indices is not None and len(indices) > 0:
        for idx in np.array(indices).flatten():
            idx = int(idx)
            x, y, w, h = boxes[idx]
            results.append({
                'class': CLASS_MAP.get(class_ids[idx], 'unknown'),
                'class_id': class_ids[idx],
                'score': round(float(scores[idx]), 4),
                'x': int(x + w / 2),
                'y': int(y + h / 2),
                'w': int(w), 'h': int(h)
            })
    return results


def detect_with(net, region, conf_thres):
    blob, orig_size = preprocess(region)
    net.setInput(blob)
    outs = net.forward()
    return postprocess(outs, orig_size, conf_thres, 0.45)


def merge_detections(lists, nms_thres=0.45):
    """多模型融合：并集后按类别 NMS 去重"""
    all_det = [d for lst in lists for d in lst]
    if not all_det:
        return []
    merged = []
    for cls in set(d['class'] for d in all_det):
        sub = [d for d in all_det if d['class'] == cls]
        boxes = [[d['x'] - d['w'] // 2, d['y'] - d['h'] // 2, d['w'], d['h']] for d in sub]
        scores = [d['score'] for d in sub]
        idx = cv2.dnn.NMSBoxes(boxes, scores, 0.1, nms_thres)
        if len(idx):
            for k in np.array(idx).flatten():
                merged.append(sub[int(k)])
    return merged


def find_best(results, class_name):
    cands = [r for r in results if r['class'] == class_name]
    if not cands:
        return None
    return max(cands, key=lambda r: r['score'])


def main():
    if len(sys.argv) < 5:
        print(json.dumps({'error': 'Usage: yolo_realtime.py <x> <y> <w> <h> [threshold] [detect_interval] [measure_interval] [collect_dir] [collect_min_score]'}), flush=True)
        sys.exit(1)

    rx, ry, rw, rh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    conf_thres = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    detect_interval = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    measure_interval = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5
    collect_dir = sys.argv[8] if len(sys.argv) > 8 else os.path.join(BASE_DIR, 'collected')
    collect_min_score = float(sys.argv[9]) if len(sys.argv) > 9 else 0.15  # 更低阈值收集更多训练样本
    capture_mode = int(sys.argv[10]) if len(sys.argv) > 10 else 0  # 0=主显示器(dxcam), 1=副显示器(mss绝对坐标)

    net = load_net()
    if net is None:
        print(json.dumps({'error': 'Model not found: ' + MODEL_PATH}), flush=True)
        sys.exit(1)
    # 可选第二模型(v8n)用于多模型融合
    net2 = None
    if os.path.exists(MODEL2_PATH):
        try:
            net2 = cv2.dnn.readNetFromONNX(MODEL2_PATH)
            net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        except Exception:
            net2 = None

    grabber = ScreenGrabber(rx, ry, rw, rh, capture_mode)
    collector = SampleCollector(collect_dir, max_samples=MAX_SAMPLES, min_score=collect_min_score)

    print(json.dumps({
        'status': 'ready',
        'region': {'x': rx, 'y': ry, 'w': rw, 'h': rh},
        'model': 'YOLOv5-WarThunder(real-trained)',
        'capture': grabber.backend,
        'classes': ['jiantou', 'huangbiao'],
        'threshold': conf_thres,
        'detect_interval': detect_interval,
        'measure_interval': measure_interval
    }), flush=True)

    latest = None
    last_detect = 0.0
    last_measure = 0.0

    try:
        while True:
            now = time.time()

            if now - last_detect >= detect_interval:
                last_detect = now
                try:
                    region = grabber.grab()
                    if region is not None and region.size > 0:
                        d1 = detect_with(net, region, conf_thres)
                        if net2 is not None:
                            d2 = detect_with(net2, region, conf_thres)
                            detections = merge_detections([d1, d2])
                        else:
                            detections = d1
                        a = find_best(detections, 'huangbiao')   # 黄标
                        b = find_best(detections, 'jiantou')     # 箭头
                        latest = (a, b, len(detections))
                        # 边测距边收集训练样本
                        saved = collector.collect(region, detections, now)
                        if saved:
                            print(json.dumps({'status': 'collected', 'total': collector.count}), flush=True)
                except Exception as e:
                    print(json.dumps({'error': str(e)}), flush=True)

            if now - last_measure >= measure_interval:
                last_measure = now
                if latest is None:
                    print(json.dumps({'status': 'searching'}), flush=True)
                else:
                    a, b, total = latest
                    if a and b:
                        dx = b['x'] - a['x']
                        dy = b['y'] - a['y']
                        pixel_dist = round((dx ** 2 + dy ** 2) ** 0.5, 1)
                        output = {
                            'status': 'measured',
                            'a': {**a, 'abs': {'x': a['x'] + rx, 'y': a['y'] + ry}},
                            'b': {**b, 'abs': {'x': b['x'] + rx, 'y': b['y'] + ry}},
                            'dx': int(abs(dx)),
                            'dy': int(abs(dy)),
                            'pixel_distance': pixel_dist,
                            'total_detected': total
                        }
                    elif a:
                        output = {'status': 'partial', 'found': 'A', 'a': a}
                    elif b:
                        output = {'status': 'partial', 'found': 'B', 'b': b}
                    else:
                        output = {'status': 'searching', 'total_detected': total}
                    print(json.dumps(output), flush=True)

            time.sleep(0.05)
    finally:
        grabber.release()
        collector.release()


if __name__ == '__main__':
    main()
