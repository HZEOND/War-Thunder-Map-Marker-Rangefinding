# WT屏幕像素测距

**War Thunder Map Marker Rangefinding**

一款免费、开源的《战争雷霆》小地图测距辅助工具。通过 AI 视觉识别小地图上的「黄标」与「坦克箭头」，自动计算目标距离，并支持边测距边采集样本、训练你自己的识别模型。

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 这是什么

在《战争雷霆》中，小地图上的 **黄标**（队友标记）与 **坦克箭头** 之间的像素距离，配合地图比例尺，可以换算成真实的游戏距离。本软件通过 **AI 目标检测** 自动识别这两个目标，实时计算距离并语音播报，替代繁琐的手动测量。

## 主要功能

### 🎯 AI 识别测距
- 框选小地图区域，自动识别「黄标」与「箭头」两个目标
- 实时计算两者中心点的像素距离，并按比例系数换算为实际距离
- 识别置信度阈值可调（0.15 ~ 0.80）

### 📐 手动像素测距
- 四种测量模式：直线 / 水平 / 垂直 / 矩形
- 放大镜辅助精确定位
- 自定义换算系数（地图单格距离 ÷ 测量像素值）

### 🧠 样本采集与 AI 训练
- 边测距边自动采集训练样本（低阈值收集弱检出）
- 人工挑选高质量样本 / 自动筛选
- 增量训练自己的 YOLOv8 模型，越练越准
- F1 不劣化保护：识别率更差的新模型永不替换好模型

### 🔊 语音播报
- 测距结果自动语音播报（Microsoft edge-tts）
- 多种内置音色可选，支持自定义语音包

### 🖥 桌面体验
- 透明置顶窗口，鼠标穿透，不干扰游戏
- 贴边自动折叠，悬停滑出
- 多显示器 + DPI 缩放自动适配
- 主题色、透明度、快捷键自定义
- 完全独立运行，无需安装 Python

### 🔄 自动更新
- 通过 GitHub Releases 检查新版本
- 发现更新红色提醒灯提示，软件内一键下载并安装

### 🔍 其他
- Statshark 玩家数据查询
- KOOK 语音开黑社区入口

## 技术架构

| 模块 | 技术 |
|------|------|
| 桌面框架 | Electron |
| 界面 | HTML5 Canvas + CSS3 |
| 目标识别 | OpenCV DNN + YOLOv5 ONNX |
| 模型训练 | Ultralytics YOLOv8 |
| 屏幕捕获 | MSS / DXcam |
| 语音合成 | Microsoft edge-tts |
| 打包分发 | PyInstaller + 7z SFX |

## 使用说明

1. 打开软件，首次使用需阅读并同意用户协议
2. 点击「手动框选」，在游戏小地图上拖拽框选识别区域，点「确认区域」
3. 点「开始 AI 识别测距」，软件自动识别黄标与箭头并测距
4. 在设置里可调整：置信阈值、语音包、透明度、主题色、快捷键

详细操作可参考软件内置的新手指引。

## 开源协议

本项目遵循 **GPL-3.0** 开源协议。

- 完全免费、永久开源，不收取任何费用
- ⚠️ 严禁任何以盈利为目的的商业行为（转售、二次收费、捆绑付费、出租售卖、收徒收费等）
- 您可以自由使用、学习、二次开发与分发

## 免责声明

1. 本软件仅通过「截取屏幕画面 + 图像识别」工作，**不读取、不修改、不注入**任何游戏内存、文件或网络数据包，**不篡改游戏数据**，不提供任何作弊功能。
2. 纯本地运行，不联网上传任何游戏数据或个人隐私。
3. 请自行确认游戏用户协议是否允许使用屏幕辅助测量工具；因使用本软件导致的账号封禁等后果由用户自行承担。

---

**一起开黑就是对作者最大的支持** 💛

---

# English

## WT Screen Pixel Ruler

**War Thunder Map Marker Rangefinding**

A free, open-source rangefinding assistant for *War Thunder*. It uses AI vision to detect the "yellow marker" and "tank arrow" on the minimap, automatically calculates the target distance, and supports collecting samples and training your own recognition model while measuring.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## What is this

In *War Thunder*, the pixel distance between the **yellow marker** (teammate ping) and the **tank arrow** on the minimap, combined with the map scale, can be converted into a real in-game distance. This tool uses **AI object detection** to automatically recognize these two targets, computes the distance in real time, and announces it by voice — replacing tedious manual measurement.

## Features

### 🎯 AI Rangefinding
- Select a minimap region to automatically detect the "yellow marker" and "arrow" targets
- Real-time pixel distance between their centers, converted to actual distance via a scale factor
- Adjustable detection confidence threshold (0.15 ~ 0.80)

### 📐 Manual Pixel Measurement
- Four measurement modes: Line / Horizontal / Vertical / Rectangle
- Magnifier-assisted precise positioning
- Custom conversion factor (grid distance ÷ measured pixel value)

### 🧠 Sample Collection & AI Training
- Auto-collects training samples while measuring (low threshold captures weak detections)
- Manual selection of high-quality samples / auto-filtering
- Incrementally train your own YOLOv8 model — the more it trains, the better it gets
- F1 non-degradation guard: a worse new model will never replace a good one

### 🔊 Voice Announcement
- Auto voice announcement of measurement results (Microsoft edge-tts)
- Multiple built-in voices, custom voice pack support

### 🖥 Desktop Experience
- Transparent always-on-top window with mouse click-through
- Auto-collapse when docked to screen edge, slide out on hover
- Multi-monitor + DPI scaling auto-adaptation
- Customizable theme color, opacity, and hotkeys
- Fully standalone — no Python installation required

### 🔄 Auto Update
- Checks for new versions via GitHub Releases
- Red indicator light on update, one-click download & install in-app

### 🔍 More
- Statshark player data lookup
- KOOK voice community entry

## Tech Stack

| Module | Technology |
|--------|------------|
| Desktop framework | Electron |
| UI | HTML5 Canvas + CSS3 |
| Object detection | OpenCV DNN + YOLOv5 ONNX |
| Model training | Ultralytics YOLOv8 |
| Screen capture | MSS / DXcam |
| Text-to-speech | Microsoft edge-tts |
| Packaging | PyInstaller + 7z SFX |

## Usage

1. Launch the app; first-time users must read and agree to the user agreement
2. Click "Select region", drag to select the minimap area in-game, then click "Confirm"
3. Click "Start AI Ranging" — the app auto-detects the marker and arrow and measures distance
4. Adjust confidence threshold, voice, opacity, theme, and hotkeys in Settings

Refer to the in-app beginner guide for detailed steps.

## License

This project is licensed under **GPL-3.0**.

- Completely free and permanently open-source — no fees, ever
- ⚠️ Any commercial / for-profit use is strictly prohibited (reselling, charging, bundling, renting, paid tutoring, etc.)
- You are free to use, learn from, modify, and redistribute it

## Disclaimer

1. This software works solely through "screen capture + image recognition". It does **not read, modify, or inject** any game memory, files, or network packets, does **not tamper with game data**, and provides no cheating functionality.
2. It runs entirely locally and uploads no game data or personal information.
3. Please verify that your game's user agreement allows on-screen measurement tools; the user assumes all consequences (e.g. account bans) of using this software.

---

**Squad up — that's the best way to support the author** 💛
