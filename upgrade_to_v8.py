# -*- coding: utf-8 -*-
"""框架升级 YOLOv5 -> YOLOv8n，带着 YOLOv5 的数据：
1. 用旧 v5 模型对真实游戏截图做伪标注（继承 v5 知识）
2. 合成数据（图标贴到真实背景，ground-truth 标注）
3. 已收集样本（collected/）
合并训练 YOLOv8n，导出 ONNX，对比择优替换。
类别顺序: 0=jiantou 1=jiantou 2=huangbiao
"""
import os, sys, shutil, random, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
V5_ONNX = os.path.join(BASE, '..', 'hb_yolo', 'WarThunder_Yellow_Mark_Rangefinder-main', 'distance', 'code', 'yolo5', 'best.onnx')
REF_IMG = os.path.join(BASE, '..', 'hb_yolo', 'WarThunder_Yellow_Mark_Rangefinder-main', 'data', 'images')
ICON_DIR = r'C:\Users\Administrator\Desktop\新建文件夹'
COLLECTED = os.path.join(BASE, 'yolo_rt', 'collected')
OUT_ONNX = os.path.join(BASE, 'wt_best.onnx')
DS = os.path.join(BASE, 'v8_upgrade_ds')
INPUT = 480

random.seed(7); np.random.seed(7)


def read_img_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def v5_detect(net, img, thr=0.3):
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT, INPUT), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()[0]
    if out.ndim == 3:
        is_v8 = out.shape[1] < out.shape[2]
        pred = out[0].T if is_v8 else out[0]
    else:
        is_v8 = out.shape[0] < out.shape[1]
        pred = out.T if is_v8 else out
    res = []
    for row in pred:
        if is_v8:
            cx, cy, bw, bh = row[0:4]; cls = row[4:]
        else:
            cx, cy, bw, bh, obj = row[0:5]; cls = row[5:]
            cls = cls * obj
        cid = int(np.argmax(cls)); sc = float(cls[cid])
        if sc > thr:
            res.append((cid, cx/W, cy/H, bw/W, bh/H))
    return res


def build():
    for sub in ['images/train', 'labels/train']:
        os.makedirs(os.path.join(DS, sub), exist_ok=True)
    idx = 0

    # 1) v5 伪标注小地图截图 + 增强（翻转/亮度）
    net = cv2.dnn.readNetFromONNX(V5_ONNX) if os.path.exists(V5_ONNX) else None
    if net and os.path.isdir(REF_IMG):
        for f in os.listdir(REF_IMG):
            if not (f.endswith('.png') or f.endswith('.jpg')):
                continue
            img = cv2.imread(os.path.join(REF_IMG, f))
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            # 整屏->裁剪小地图区域(按1080p比例)；小图(小地图)直接用
            if W0 > 1000:
                x = int(W0 * 1462 / 1920); y = int(H0 * 622 / 1080)
                w = int(W0 * 456 / 1920); h = int(H0 * 456 / 1080)
                img = img[y:y+h, x:x+w]
            dets = v5_detect(net, img)
            variants = [(img, dets)]
            # 水平翻转
            flip = cv2.flip(img, 1)
            fdets = [(c, 1.0-cx, cy, bw, bh) for (c, cx, cy, bw, bh) in dets]
            variants.append((flip, fdets))
            # 亮度扰动
            jit = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
            variants.append((jit, dets))
            for vi, (vimg, vdets) in enumerate(variants):
                name = 'real_%04d_%d' % (idx, vi)
                cv2.imwrite(os.path.join(DS, 'images', 'train', name + '.jpg'), vimg)
                with open(os.path.join(DS, 'labels', 'train', name + '.txt'), 'w') as fo:
                    for d in vdets:
                        fo.write('%d %.6f %.6f %.6f %.6f\n' % d)
            idx += 1

    # 2) 合成数据：图标贴真实背景（图标缺失则跳过）
    arrow = yellow = None
    for cand in (os.path.join(BASE, 'arrow2.png'), os.path.join(BASE, 'yolo_rt', 'arrow2.png')):
        if os.path.exists(cand):
            arrow = cv2.imread(cand)
    for cand in (os.path.join(BASE, 'yellow2.png'), os.path.join(BASE, 'yolo_rt', 'yellow2.png')):
        if os.path.exists(cand):
            yellow = cv2.imread(cand)
    bgs = [cv2.imread(os.path.join(REF_IMG, f)) for f in os.listdir(REF_IMG)
           if f.endswith(('.png', '.jpg'))] if os.path.isdir(REF_IMG) else []
    bgs = [b for b in bgs if b is not None]
    if arrow is not None and yellow is not None and bgs:
        for i in range(300):
            bg = random.choice(bgs).copy()
            H, W = bg.shape[:2]
            lines = []
            for icon, cid in ((arrow, 0), (yellow, 2)):
                for _ in range(random.randint(1, 2)):
                    s = random.uniform(0.4, 1.6)
                    iw, ih = int(icon.shape[1]*s), int(icon.shape[0]*s)
                    if iw < 5 or ih < 5 or iw >= W or ih >= H:
                        continue
                    r = cv2.resize(icon, (iw, ih))
                    x = random.randint(0, W-iw); y = random.randint(0, H-ih)
                    bg[y:y+ih, x:x+iw] = r
                    lines.append('%d %.6f %.6f %.6f %.6f' % (cid, (x+iw/2)/W, (y+ih/2)/H, iw/W, ih/H))
            name = 'syn_%04d' % idx; idx += 1
            cv2.imwrite(os.path.join(DS, 'images', 'train', name + '.jpg'), bg)
            with open(os.path.join(DS, 'labels', 'train', name + '.txt'), 'w') as fo:
                fo.write('\n'.join(lines))

    # 3) 已收集样本
    if os.path.isdir(COLLECTED):
        for f in os.listdir(COLLECTED):
            if f.endswith('.jpg'):
                shutil.copy(os.path.join(COLLECTED, f), os.path.join(DS, 'images', 'train', 'col_' + f))
                t = os.path.join(COLLECTED, os.path.splitext(f)[0] + '.txt')
                if os.path.exists(t):
                    shutil.copy(t, os.path.join(DS, 'labels', 'train', 'col_' + os.path.splitext(f)[0] + '.txt'))

    n = len([f for f in os.listdir(os.path.join(DS, 'images', 'train'))])
    with open(os.path.join(DS, 'data.yaml'), 'w', encoding='utf-8') as fo:
        fo.write('path: %s\ntrain: images/train\nval: images/train\nnc: 3\nnames: [jiantou, jiantou2, huangbiao]\n' % os.path.abspath(DS))
    print(json.dumps({'status': 'dataset', 'count': n}), flush=True)
    return n


def train():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.train(data=os.path.join(DS, 'data.yaml'), epochs=30, imgsz=480, batch=8, device='cpu', verbose=False)
    best_pt = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    m2 = YOLO(best_pt)
    m2.export(format='onnx', imgsz=480, simplify=True)
    exported = os.path.join(model.trainer.save_dir, 'weights', 'best.onnx')
    print(json.dumps({'status': 'exported', 'path': exported}), flush=True)
    return exported


if __name__ == '__main__':
    n = build()
    if n < 5:
        print(json.dumps({'status': 'error', 'msg': '数据不足'}), flush=True)
        sys.exit(1)
    exported = train()
    print(json.dumps({'status': 'done', 'exported': exported}), flush=True)
