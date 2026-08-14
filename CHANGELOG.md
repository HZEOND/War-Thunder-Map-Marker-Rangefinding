# 更新日志 (CHANGELOG)

## v1.9.2 — 识别精确对齐 + 训练闭环修复（2026-08-13）

### 🔍 识别坐标精确对齐
- **DPI 缩放修复**：识别区域框选用 DIP(逻辑像素)、识别引擎截图用物理像素，现按 scaleFactor 换算，高 DPI 屏幕(125%/150%)识别区域不再错位
- **多显示器支持**：副显示器框选改用 mss 绝对屏幕坐标（主显示器保持 dxcam 高性能），重新打包识别引擎 exe

### 🧠 训练进度条真正显示
- 修复 ANSI 转义序列(`\u001b[K`)导致进度正则匹配失败
- 修复 `\r` 覆盖式输出导致进度挤在一行无法解析
- 进度条移到面板顶部、加大加亮(16px + 金色)，显示总进度百分比

### 📈 训练回退根因修复
- `pickPython` 加系统 Python 候选(Anaconda)，避免走含 bug 的旧版 exe
- `_count_detections` 改用 onnxruntime(OpenCV 5.0 读不了旧 ONNX)
- **关键**：修复 v5 模型检出数虚高(66920)，正确用 obj × class_score

### 🔄 训练后模型真正顶替识别
- 修复输出路径断链：训练写到 yolo_rt/ 旧路径、识别引擎读 wt_best.onnx → 改为直接覆盖 wt_best.onnx
- 修复临时目录模型 mtime 判断，新模型能自动刷新到运行时

### ♾ 增量训练持久化
- 训练起点优先 `wt_best_last.pt`(上次成果)，实现越练越准
- 训练成功后保存 best.pt 到应用目录，成果持久不丢

### 🛡 识别率不劣化保护
- accept 从「检出数≥60%」改为「新模型 F1 ≥ 旧模型 F1」(旧模型不可用则 ≥70%)
- 用 NMS+IoU 精确评估 F1，识别率更差的模型永不替换好模型

### 📁 采集样本持久化
- 新增应用根目录「采集样本」文件夹，人工挑选样本自动同步备份，可自行管理

### ⏸ 训练与采集联动
- 训练开始自动暂停样本采集(stdin pause)、结束自动恢复(resume)，避免污染训练集

### 🖱 人工筛选体验
- 左键拖拽框选批量勾选样本(Shift/Alt 取消)
- 滚轮上下滚动查看所有样本
- 点击图片查看原图

### 🧹 UI 精简 + ⚡ 内存优化
- 移除 AI 视野缩略图(省面板空间 + 减少 CPU/IPC 开销)
- V8 堆限制 256MB、单渲染进程、禁用 GPU shader 缓存

## v1.9.0 — 训练可视化 + 临时目录统一（2026-08-13）

### 🆕 新功能
- **训练进度条**：训练时实时显示 epoch 进度 + 百分比进度条（解析 ultralytics 日志）
- **回退样本标记**：训练回退后在人工筛选窗口每张图右上角打 ✕ 标记，提示删除异常样本
- **样本大图预览**：点击缩略图弹出原图查看（95vw/95vh 自适应），点击关闭
- **打开样本文件夹**：人工筛选窗口新增「📂 打开文件夹」按钮，自行管理样本文件

### 🔧 Bug 修复
- **修复停止 AI 识别后进程残留**：`.kill()` 只杀父进程，改用 `taskkill /pid /T /F` 杀整个进程树
- **修复训练后 yolov8n.pt 损坏**：ultralytics 检测损坏后重下 GitHub 超时 → 加 `getsize>6MB` 校验 + 回退 `yolov8n.yaml` 从头训练
- **修复训练 SyntaxError**：retrain-ai 三元表达式漏 else 分支 → 重构 spawn 逻辑
- **修复模型验证过严**：`new>=cur` 放宽到 `new>=cur*0.6` 或旧模型不可用，小数据集不再被全拒
- **修复人工筛选无法滚动**：grid 加 `min-height:0`（flexbox 滚动必需）
- **删除样本进回收站**：`fs.unlinkSync` → `shell.trashItem()`，可恢复
- **修复训练污染安装目录**：ultralytics runs/ 输出到 CWD（50M 垃圾），重定向到临时目录

### 🗂 临时目录统一
- 新增 `getRtDir()` 统一根目录 `%TEMP%/pixel_ruler_rt/`
- 结构：`runtime/`(运行时缓存) + `collected/`(样本) + `clean/`(训练集) + `retrain_ds/`(数据集) + `train_runs/`(训练输出)

### 🎯 训练参数
- epochs 10→50，imgsz 480→640（帮助检测微小目标）
- 训练优先走系统 Python（Anaconda 有 ultralytics），exe 待重建

## v1.8.1 — 流畅度优化（2026-08-12）

### ⚡ 性能
- 鼠标穿越面板检测 80ms→100ms 节流，rect 缓存 300ms→500ms，减少 30% IPC 调用
- 修复非测量模式 mousemove 不更新穿透状态，面板按钮始终可点击
- 启动延时保护 2s 确保元素加载完毕

### 🐛 修复
- 边缘吸附折叠后测量值框消失（distance-display HTML 移出 panel-body）
- Statshark 搜索改为独立 BrowserWindow（绕过 iframe Cloudflare 限制）
- 标题栏按钮无法点击（mousemove guard 条件修复）

## v1.8.0 — 重大更新（2026-08-12）

### 🆕 新功能
- **筛选样本升级**：两种模式 —「🤖 自动筛选」规则+置信度一键筛选；「👁 人工挑选（提升更大）」直接打开挑选窗口逐张勾选高质量样本上传训练
- **AI 视野预览**：识别时面板底部实时显示框选区域的缩略图，一眼看出区域是否正确
- **区域错位警告**：连续10秒无检出自动红色提示「区域可能错位或游戏未运行」
- **边缘吸附**：拖动面板到屏幕边缘自动折叠为最小状态（仅留测量值），鼠标悬停滑出恢复完整面板
- **框选覆盖任务栏**：手动框选时窗口临时提升到 floating 层级，可框到任务栏附近区域
- **日志合并**：导出日志到桌面按钮内嵌到日志查看器底部
- **样本计数实时刷新**：采集成功立即更新数字，不再等5秒轮询
- **响应速度优化**：截图间隔 0.7s→0.20s，测距输出 1.0s→0.35s，OMP_NUM_THREADS=4 多核加速

### 🔧 Bug 修复
- **修复 YOLO 识别无法启动**：yolo_rt.exe 缺少 _internal/python313.dll，改用自包含 yolo_realtime.exe 单文件版
- **修复 OpenCV 中文路径崩溃**：cv2.dnn.readNetFromONNX / cv2.imwrite 均无法处理中文路径 → 运行时+模型+样本统一复制到 `%TEMP%/pixel_ruler_rt/` 英文目录
- **修复「开始 AI 识别测距」无法点击**：移除 disabled 默认全屏识别
- **修复 SyntaxError: duplicate const py**：优化编辑时残留旧 spawn 行导致主进程崩溃
- **修复采集停止**：collect_dir 跟随 prepTempRuntime 改到英文临时目录，训练/筛选/管理路径统一
- **修复 V8 置信度异常**：V8 模型输原始 logits 导致 UI 显示几万%，main.js 回调中自动 sigmoid 归一化
- **修复训练热重载断联**：retrain-ai 完成后热重启识别进程无 stdout handler，重构为 setupRealtimeProcess() 统一管理

### 🎯 模型/训练
- **采集阈值优化**：min_score 0.5→0.15（重新编译 yolo_realtime.exe），弱检出也能存为训练数据
- **识别引擎**：YOLOv5 (28MB wt_best.onnx) 主识别，YOLOv8 (12MB wt_best_v8.onnx) 备用/训练目标
- **导入参考项目**：黄标测距 yolo_ref/（含 arrow.yaml 训练配置、多分辨率 Map 模板、YOLOv5 代码库）
- **训练独立化**：retrain_yolo.exe (901MB) + auto_filter_samples.exe (274MB) PyInstaller 打包，不依赖系统 Python

### 📦 工程
- **安装包**：7z SFX 自解压，免 UAC 用户级安装，Setup.exe 1482 MB
- **安装包解决中文路径**：整个运行时自动缓存到 `%TEMP%/pixel_ruler_rt/`
- **版本**：1.7.1 → 1.8.0
- **开源架构**：Electron + YOLOv5 ONNX（OpenCV DNN），支持 v5/v8 双模型，边测距边训练在线闭环

## v1.7.2
- **修复 YOLO 识别无法启动**（yolo_rt.exe 缺少 _internal/python313.dll，改为自包含 yolo_realtime.exe 单文件版）
- **修复 OpenCV 中文路径 ONNX 模型加载失败**（main.js 启动时自动将 exe+模型复制到英文临时目录，绕过 cv2.dnn C 层的中文路径限制）
- **切换到 YOLOv8 识别**（wt_best_v8.onnx → wt_best.onnx 作为主模型，v5 备份为 wt_best_v5.onnx）
- **导入黄标测距参考项目**（yolo_ref/，含训练配置 arrow.yaml、YOLOv5 代码库、多分辨率模板）
- **修复「开始 AI 识别测距」按钮无法点击**（默认全屏识别，无需先框选区域）
- 同步更新桌面绿色版和安装包

## v1.5.2
- 截图底层由 mss 升级为 **DXcam**（Desktop Duplication，高性能、低延迟、不丢帧），保留 mss 回退
- 新增**打包前文件完整性校验**，文件缺失时中止打包，避免残缺包
- 开源同步：更新 README 与 Release
- 重编译 yolo_rt.exe 内置 DXcam

## v1.5.1
- 修复切屏回来 / 测量模式无法点击的问题（焦点与点击穿透重同步）
- 设置与公告改为左侧滑出抽屉，不影响主界面大小
- "YOLO目标检测测距" 改名 "AI检测"，删除 "YOLO已训练" 冗余行
- "安装公告" 改名 "公告栏"，公告倒序排列
- 新增顶部 🎮 KOOK 开黑入口（悬停/固定弹窗 + 二维码 + 链接）
- 新增设置内"文件完整性校验"
- 按钮 hover / active 反馈

## v1.5.0
- 优化 UI 大小（实际测量值框自适应宽度）
- 新增新用户使用指引（6 步）
- 应用/托盘图标自定义
- 关闭对话框（最小化托盘 / 退出 + 记住选择）

## v1.0.0
- 首个版本：手动测距 + AI 识别测距 + 语音播报
