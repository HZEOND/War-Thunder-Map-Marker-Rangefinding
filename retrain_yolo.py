# -*- coding: utf-8 -*-
"""增量训练：用边测距边收集的样本微调 YOLOv8，导出 ONNX 热更新模型
用法: python retrain_yolo.py <collected_dir> <out_onnx> [epochs]
环境: 优先内置 trainer/python/python.exe，回退当前/系统 Python
"""
import os, sys, shutil, json

# 修复 OpenMP 运行时冲突（torch 与 ultralytics/opencv 各带 libiomp5md.dll）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '4')

BASE = os.path.dirname(os.path.abspath(__file__))

def _maybe_reexec_embedded():
    if os.environ.get('RETRAIN_NO_REEXEC'):
        return
    candidates = [
        os.path.join(BASE, 'trainer', 'python', 'python.exe'),
        os.path.join(os.path.dirname(BASE), 'trainer', 'python', 'python.exe'),
    ]
    for py in candidates:
        if os.path.exists(py) and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(py):
            env = dict(os.environ, RETRAIN_NO_REEXEC='1')
            os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], env)

_maybe_reexec_embedded()

def detect_env():
    try:
        import ultralytics  # noqa
        return {'ok': True, 'python': sys.executable}
    except Exception as e:
        return {'ok': False, 'err': str(e)}

def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description='YOLOv8n 增量训练（带 P0-P5 改进）')
    p.add_argument('collected', nargs='?', default='collected', help='收集样本目录')
    p.add_argument('out_onnx', nargs='?', default='wt_best.onnx', help='输出 ONNX 路径')
    # P0 数据增强
    p.add_argument('--augment', action='store_true', help='开启数据增强')
    p.add_argument('--hsv_h', type=float, default=0.015)
    p.add_argument('--hsv_s', type=float, default=0.7)
    p.add_argument('--hsv_v', type=float, default=0.4)
    p.add_argument('--degrees', type=float, default=10.0)
    p.add_argument('--translate', type=float, default=0.1)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--flipud', type=float, default=0.0)
    p.add_argument('--fliplr', type=float, default=0.5)
    p.add_argument('--mosaic', type=float, default=0.5)
    # P1/P2
    p.add_argument('--pseudo_label', action='store_true', help='v5 伪标签自训练')
    p.add_argument('--negative_samples', action='store_true', help='加入负样本')
    # P3 微调策略
    p.add_argument('--freeze', type=int, default=0, help='冻结前 N 层')
    p.add_argument('--lr0', type=float, default=0.01)
    # 训练超参
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=480)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu')
    return vars(p.parse_args(argv))


def _pseudo_label_v5(collected, ds):
    """用 v5 参考模型(ONNX)为 collected 图片生成/补充伪标签"""
    import cv2, numpy as np
    v5 = os.path.join(BASE, '..', 'hb_yolo', 'WarThunder_Yellow_Mark_Rangefinder-main',
                      'distance', 'code', 'yolo5', 'best.onnx')
    if not os.path.exists(v5):
        return 0
    net = cv2.dnn.readNetFromONNX(v5)
    n = 0
    for f in os.listdir(collected):
        if not f.endswith('.jpg'):
            continue
        lbl = os.path.join(ds, 'labels', 'train', f.replace('.jpg', '.txt'))
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            continue
        img = cv2.imread(os.path.join(collected, f))
        if img is None:
            continue
        H, W = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (480, 480), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()[0]
        if out.ndim == 3:
            is_v8 = out.shape[1] < out.shape[2]; pred = out[0].T if is_v8 else out[0]
        else:
            is_v8 = out.shape[0] < out.shape[1]; pred = out.T if is_v8 else out
        lines = []
        for row in pred:
            if is_v8:
                cx, cy, bw, bh = row[0:4]; cls = row[4:]; sc = float(np.max(cls))
            else:
                cx, cy, bw, bh, obj = row[0:5]; cls = row[5:]; sc = float(obj*np.max(cls))
            cid = int(np.argmax(cls))
            if sc > 0.3:
                lines.append(f"{cid} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}")
        with open(lbl, 'w') as fo:
            fo.write('\n'.join(lines))
        n += 1
    return n


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print(json.dumps({'status': 'env', **detect_env()}), flush=True)
        return
    a = _parse_args(sys.argv[1:])
    collected = a['collected']; out_onnx = a['out_onnx']; epochs = a['epochs']

    imgs = [f for f in os.listdir(collected) if f.endswith('.jpg')]
    print(json.dumps({'status': 'info', 'samples': len(imgs), 'env': detect_env()['ok']}), flush=True)
    if len(imgs) < 5:
        print(json.dumps({'status': 'error', 'msg': '样本不足(<5)，继续测距收集后再训练'}), flush=True)
        sys.exit(1)

    ds = os.path.join(os.path.dirname(collected) or '.', 'retrain_ds')
    for sub in ['images/train', 'labels/train']:
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    for f in imgs:
        base = os.path.splitext(f)[0]
        shutil.copy(os.path.join(collected, f), os.path.join(ds, 'images', 'train', f))
        txt = base + '.txt'
        if os.path.exists(os.path.join(collected, txt)):
            shutil.copy(os.path.join(collected, txt), os.path.join(ds, 'labels', 'train', txt))

    # P1 伪标签自训练
    if a['pseudo_label']:
        _pseudo_label_v5(collected, ds)
    # P2 负样本挖掘（背景-only，空标签）
    if a['negative_samples']:
        neg = os.path.join(collected, 'negatives')
        if os.path.isdir(neg):
            for f in os.listdir(neg):
                if f.endswith('.jpg'):
                    shutil.copy(os.path.join(neg, f), os.path.join(ds, 'images', 'train', 'neg_' + f))
                    open(os.path.join(ds, 'labels', 'train', 'neg_' + f.replace('.jpg', '.txt')), 'w').close()

    yaml_path = os.path.join(ds, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(ds)}\ntrain: images/train\nval: images/train\nnc: 3\nnames: [jiantou, jiantou2, huangbiao]\n")

    from ultralytics import YOLO
    print(json.dumps({'status': 'training', 'epochs': epochs, 'augment': a['augment']}), flush=True)
    # 增量训练：优先用上次训练的 best.pt 作为起点（越练越准），否则用 yolov8n.pt 预训练
    last_pt = os.path.join(BASE, 'wt_best_last.pt')
    pt_candidates = [
        last_pt,   # 上次训练成果（增量起点）
        os.path.join(BASE, 'yolov8n.pt'),
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'yolov8n.pt'),
        'yolov8n.pt'
    ]
    base_model = next((p for p in pt_candidates if os.path.exists(p) and os.path.getsize(p) > 6000000), None)
    if base_model:
        print(json.dumps({'status': 'info', 'msg': f'Loading from {base_model}'}), flush=True)
        try:
            model = YOLO(base_model)
        except Exception:
            print(json.dumps({'status': 'info', 'msg': 'Loading failed, building from scratch'}), flush=True)
            model = YOLO('yolov8n.yaml')
    else:
        print(json.dumps({'status': 'info', 'msg': 'No valid base model found, building from scratch'}), flush=True)
        model = YOLO('yolov8n.yaml')
    import tempfile
    # 训练输出(runs/)统一放到临时目录，避免污染安装目录
    train_project = os.path.join(tempfile.gettempdir(), 'pixel_ruler_rt', 'train_runs')
    train_kwargs = dict(data=yaml_path, epochs=epochs, imgsz=a['imgsz'], batch=a['batch'],
                        device=a['device'], verbose=True, lr0=a['lr0'], project=train_project)
    if a['freeze'] > 0:
        train_kwargs['freeze'] = a['freeze']
    if a['augment']:   # P0 数据增强
        train_kwargs.update(hsv_h=a['hsv_h'], hsv_s=a['hsv_s'], hsv_v=a['hsv_v'],
                            degrees=a['degrees'], translate=a['translate'], scale=a['scale'],
                            flipud=a['flipud'], fliplr=a['fliplr'], mosaic=a['mosaic'])
    model.train(**train_kwargs)

    best_pt = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    m2 = YOLO(best_pt)
    m2.export(format='onnx', imgsz=480, simplify=True)
    exported = os.path.join(model.trainer.save_dir, 'weights', 'best.onnx')

    # 对比择优：用 NMS+IoU 精确评估两个模型的 F1 识别率
    tmp_new = out_onnx + '.new'
    shutil.copy(exported, tmp_new)
    cur_f1 = _eval_f1(out_onnx, collected)
    new_f1 = _eval_f1(tmp_new, collected)
    # 相对阈值：新模型 F1 ≥ 旧模型 F1 才替换（保证不劣化）
    # 旧模型不可用时（首次训练/加载失败），门槛放宽到 70%
    if cur_f1 < 0:
        accept = new_f1 >= 70.0
        reason = f'旧模型不可用，新模型 F1={new_f1:.1f}% ≥ 70%'
    else:
        accept = new_f1 >= cur_f1
        reason = f'新模型 F1={new_f1:.1f}% ≥ 旧模型 {cur_f1:.1f}%'
    if accept:
        if os.path.exists(out_onnx):
            shutil.copy(out_onnx, out_onnx + '.bak')
        shutil.copy(tmp_new, out_onnx)
        # 保存 best.pt 到应用目录，作为下次增量训练的起点（越练越准）
        try:
            last_pt_path = os.path.join(BASE, 'wt_best_last.pt')
            if os.path.exists(best_pt):
                shutil.copy(best_pt, last_pt_path)
                print(json.dumps({'status': 'info', 'msg': '已保存增量训练起点 wt_best_last.pt'}), flush=True)
        except Exception as e:
            print(json.dumps({'status': 'info', 'msg': f'保存增量权重失败: {e}'}), flush=True)
        os.remove(tmp_new)
        print(json.dumps({'status': 'done', 'out': out_onnx, 'samples': len(imgs),
                          'cur_f1': cur_f1, 'new_f1': new_f1}), flush=True)
    else:
        if os.path.exists(tmp_new):
            os.remove(tmp_new)
        print(json.dumps({'status': 'rejected',
                          'msg': f'{reason}；已保留旧模型，继续采集样本后再训练',
                          'samples': len(imgs), 'cur_f1': cur_f1, 'new_f1': new_f1}), flush=True)


def _iou(a, b):
    """计算两个框 [x1,y1,x2,y2] 的 IoU"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


def _eval_f1(onnx_path, collected):
    """用 NMS + IoU 匹配精确评估模型 F1 识别率（百分比）；加载失败返回 -1
    优先用 onnxruntime（兼容 OpenCV 5.0 无法加载旧 ONNX 的问题）"""
    try:
        import cv2, numpy as np
        import onnxruntime as ort
        jpgs = [f for f in os.listdir(collected) if f.endswith('.jpg')]
        if not jpgs:
            return 0.0
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        in_name = sess.get_inputs()[0].name
        tp = fp = fn = 0
        for f in jpgs:
            img = cv2.imread(os.path.join(collected, f))
            if img is None:
                continue
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (480, 480), swapRB=True, crop=False)
            out = sess.run(None, {in_name: blob})[0]
            if out.ndim == 3:
                is_v8 = out.shape[1] < out.shape[2]
                pred = out[0].T if is_v8 else out[0]
            else:
                is_v8 = out.shape[0] < out.shape[1]
                pred = out.T if is_v8 else out
            boxes, scores = [], []
            for row in pred:
                cx, cy, bw, bh = row[0:4]
                if is_v8:
                    score = float(np.max(row[4:]))
                else:
                    score = float(row[4]) * float(np.max(row[5:]))
                if score < 0.3:
                    continue
                x1 = (cx - bw / 2) * w / 480; y1 = (cy - bh / 2) * h / 480
                x2 = (cx + bw / 2) * w / 480; y2 = (cy + bh / 2) * h / 480
                boxes.append([x1, y1, x2, y2]); scores.append(score)
            dets = []
            if boxes:
                idx = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.5)
                if len(idx) > 0:
                    dets = [boxes[int(i)] for i in np.array(idx).flatten()]
            # 读 ground truth 标签
            gts = []
            txt = os.path.join(collected, f.replace('.jpg', '.txt'))
            if os.path.exists(txt):
                with open(txt) as fio:
                    for line in fio:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        _, cx, cy, bw, bh = map(float, parts[:5])
                        gts.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                    (cx + bw / 2) * w, (cy + bh / 2) * h])
            matched = [False] * len(gts)
            for d in dets:
                best_iou, best_j = 0, -1
                for j, g in enumerate(gts):
                    if matched[j]:
                        continue
                    i = _iou(d, g)
                    if i > best_iou:
                        best_iou, best_j = i, j
                if best_iou >= 0.5 and best_j >= 0:
                    matched[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(1 for m in matched if not m)
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec) * 100
    except Exception as e:
        print(json.dumps({'status': 'info', 'msg': f'F1评估异常: {e}'}), flush=True)
        return -1

if __name__ == '__main__':
    main()
