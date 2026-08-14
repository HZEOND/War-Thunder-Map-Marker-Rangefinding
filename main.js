// main.js — 基于 2.txt 透明置顶窗口，完整测距应用
// 快捷键避免ESC冲突，使用 Ctrl+Shift+ 安全组合
const { app, BrowserWindow, screen, ipcMain, globalShortcut, desktopCapturer, Tray, Menu, nativeImage, shell, net } = require('electron')
const path = require('path')
const fs = require('fs')
const { spawn } = require('child_process')

// ===== 内存优化：限制 Chromium/V8 内存占用（必须在 app ready 前设置）=====
app.commandLine.appendSwitch('js-flags', '--max-old-space-size=256')   // 限制渲染进程 V8 堆 256MB
app.commandLine.appendSwitch('renderer-process-limit', '1')             // 单渲染进程，避免多进程内存
app.commandLine.appendSwitch('disable-gpu-shader-disk-cache')           // 禁用 GPU shader 磁盘缓存
app.commandLine.appendSwitch('disable-http-cache')                      // 禁用 HTTP 缓存（本应用不依赖）
app.commandLine.appendSwitch('disable-features', 'TranslateUI')         // 关闭翻译相关

// ===== 单实例锁：重复启动时聚焦已有窗口，不打开多个 =====
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) win.restore()
      win.show()
      win.focus()
    }
  })
}

let win = null

// ===== 自动更新（GitHub Releases）=====
// GitHub 仓库地址（检查更新用）：用户名/仓库名
const UPDATE_REPO = 'HZEOND/War-Thunder-Map-Marker-Rangefinding'
const UPDATE_API = `https://api.github.com/repos/${UPDATE_REPO}/releases/latest`

// 版本号比较：返回 a>b ? 1 : a<b ? -1 : 0（忽略 v 前缀）
function compareVersions(a, b) {
  const pa = String(a || '').replace(/^v/i, '').split('.').map(n => parseInt(n, 10) || 0)
  const pb = String(b || '').replace(/^v/i, '').split('.').map(n => parseInt(n, 10) || 0)
  const len = Math.max(pa.length, pb.length)
  for (let i = 0; i < len; i++) {
    const x = pa[i] || 0, y = pb[i] || 0
    if (x > y) return 1
    if (x < y) return -1
  }
  return 0
}

// ===== 使用日志 =====
function logPath() { return path.join(app.getPath('userData'), 'pixel-ruler.log') }
function log(msg) {
  try {
    const line = '[' + new Date().toLocaleString('zh-CN') + '] ' + msg + '\n'
    fs.appendFileSync(logPath(), line)
    // 限制日志大小（>1MB 截断）
    const st = fs.statSync(logPath())
    if (st.size > 1024 * 1024) {
      const content = fs.readFileSync(logPath(), 'utf-8')
      fs.writeFileSync(logPath(), content.slice(-500 * 1024))
    }
  } catch (e) {}
}

// 解析 Python 脚本真实路径（打包后 .py 被解包到 app.asar.unpacked）
function getPyPath(name) {
  let p = path.join(__dirname, name)
  if (p.includes('app.asar')) {
    p = p.replace('app.asar', 'app.asar.unpacked')
  }
  return p
}

// 优先用 PyInstaller 编译的 exe（完全独立，无需 Python），否则回退 python+脚本
function getRuntime(exeRelPath, pyName, scriptArgs) {
  const exePath = getPyPath(exeRelPath)
  if (fs.existsSync(exePath)) {
    return { cmd: exePath, args: scriptArgs }
  }
  return { cmd: 'python', args: [getPyPath(pyName), ...scriptArgs] }
}

// 统一临时根目录：所有运行时临时文件都放在这一个文件夹下
function getRtDir() {
  const os = require('os')
  const d = path.join(os.tmpdir(), 'pixel_ruler_rt')
  fs.mkdirSync(d, { recursive: true })
  return d
}

// 解决 OpenCV cv2.dnn.readNetFromONNX 无法读取中文路径的问题
// 首次运行时将 yolo_realtime.exe 和 ONNX 模型复制到英文临时目录的 runtime/ 子目录
function prepTempRuntime() {
  const tempDir = path.join(getRtDir(), 'runtime')
  fs.mkdirSync(tempDir, { recursive: true })
  const srcExe = getPyPath('yolo_realtime.exe')
  const dstExe = path.join(tempDir, 'yolo_realtime.exe')
  const needRefresh = !fs.existsSync(dstExe) || fs.statSync(srcExe).mtimeMs > fs.statSync(dstExe).mtimeMs
  if (needRefresh) {
    try { fs.copyFileSync(srcExe, dstExe) } catch (e) {}
    log('YOLO 运行时已缓存到临时目录（解决中文路径问题）')
  }
  for (const m of ['wt_best.onnx', 'wt_best_v8.onnx']) {
    const srcM = getPyPath(m)
    const dstM = path.join(tempDir, m)
    // 每个模型独立判断 mtime，训练后新模型能自动刷新到临时目录
    const modelRefresh = fs.existsSync(srcM) && (!fs.existsSync(dstM) || fs.statSync(srcM).mtimeMs > fs.statSync(dstM).mtimeMs)
    if (modelRefresh) {
      try { fs.copyFileSync(srcM, dstM) } catch (e) {}
    }
  }
  return dstExe
}

// 样本采集/训练统一存到英文临时目录，避免 cv2.imwrite 中文路径写入失败
function getCollectDir() {
  const d = path.join(getRtDir(), 'collected')
  fs.mkdirSync(d, { recursive: true })
  return d
}

// 持久样本目录：应用根目录下的「采集样本」文件夹，方便用户自行查看/删除
function getPersistSamplesDir() {
  const appRoot = path.dirname(path.dirname(__dirname))
  const d = path.join(appRoot, '采集样本')
  fs.mkdirSync(d, { recursive: true })
  return d
}

let shortcutConfig = {
  exit: 'CommandOrControl+Shift+X',
  hide: 'CommandOrControl+Shift+H',
  clear: 'CommandOrControl+Shift+C'
}

function createWindow(targetDisplay) {
  const displays = screen.getAllDisplays()
  const primaryDisplay = screen.getPrimaryDisplay()
  const useDisplay = targetDisplay || primaryDisplay

  win = new BrowserWindow({
    title: 'WT屏幕像素测距',
    x: useDisplay.bounds.x, y: useDisplay.bounds.y,
    width: useDisplay.size.width, height: useDisplay.size.height,
    transparent: true, frame: false, alwaysOnTop: true,
    resizable: false, skipTaskbar: false, hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false,
      webSecurity: false   // 允许 iframe 加载跨域页面
    }
  })

  // 首次运行（未完成指引）不启用点击穿透，保证新用户可点击欢迎弹窗与指引
  let firstRun = true
  try {
    const settingsFile = path.join(app.getPath('userData'), 'pixel-ruler-settings.json')
    if (fs.existsSync(settingsFile)) {
      const s = JSON.parse(fs.readFileSync(settingsFile, 'utf-8'))
      firstRun = !s.guideDone
    }
  } catch (e) { firstRun = true }
  win.setIgnoreMouseEvents(firstRun ? false : true, { forward: true })
  win.setAlwaysOnTop(true, 'screen-saver')
  win.loadFile('index.html')

  // 拦截 Statshark 响应头，剥离 X-Frame-Options 以允许 iframe 嵌入
  win.webContents.session.webRequest.onHeadersReceived(
    { urls: ['*://statshark.net/*'] },
    (details, callback) => {
      const responseHeaders = Object.assign({}, details.responseHeaders)
      delete responseHeaders['x-frame-options']
      delete responseHeaders['content-security-policy']
      callback({ responseHeaders })
    }
  )

  win.webContents.on('did-finish-load', () => {
    win.webContents.send('displays-info', displays.map(d => ({
      id: d.id, bounds: d.bounds, size: d.size,
      scaleFactor: d.scaleFactor, isPrimary: d.id === primaryDisplay.id
    })))
    win.webContents.send('shortcut-config', shortcutConfig)
  })
}

function registerShortcuts() {
  globalShortcut.unregisterAll()
  if (shortcutConfig.exit) globalShortcut.register(shortcutConfig.exit, () => app.quit())
  if (shortcutConfig.hide) globalShortcut.register(shortcutConfig.hide, () => {
    if (win) { if (win.isVisible()) win.hide(); else win.show() }
  })
  if (shortcutConfig.clear) globalShortcut.register(shortcutConfig.clear, () => {
    if (win) win.webContents.send('clear-measurement')
  })
}

// ===== 系统托盘：最小化时隐藏到右下角托盘 =====
let tray = null
const TRAY_ICON_B64 = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAABLUlEQVRIDbXB0a0qAQhF0UMP8Lu7oWq64ZcmXmJigrn6nHF0LXPQL5mDfskc9EvmoF8yB33PVEeixRz0JVOtm0h0Zw76hqnWEoluzEGXTbWWSHRnDrpmqrVEosUcdMFUa4lEj8xBn5pqLZHoD3PQR6ZaSyR6xhx03lRriUQvmINOmmotkUx1JHrGHHTGVGuJRDdTHYn+MAcdNtVaItE75qBjplpLJDrAHHTAVGuJRMeYg96Zai2RTHUkOsAc9F9TrSUS3Ux1JHrHHPTaVGuJRCeZg16Yai2R6Dxz0DNTrSUSfcQc9MdUa4lkqiPReeagR1OtJRLdTHUkOskctEy1lkh0jTnobqq1RKLLzEHLVOsmEn2DOejRVEcy1ZHoMnPQC1Mdia4xB/2SOeiX/gGq4W0hVakNrQAAAABJRU5ErkJggg=='

function createTray() {
  const iconPath = getPyPath('icon32.png')
  let icon
  if (fs.existsSync(iconPath)) {
    icon = nativeImage.createFromPath(iconPath)
  } else {
    icon = nativeImage.createFromDataURL('data:image/png;base64,' + TRAY_ICON_B64)
  }
  tray = new Tray(icon)
  tray.setToolTip('WT屏幕像素测距')
  const contextMenu = Menu.buildFromTemplate([
    { label: '显示窗口', click: () => { if (win) { win.show(); win.focus() } } },
    { type: 'separator' },
    { label: '退出', click: () => { app.quit() } }
  ])
  tray.setContextMenu(contextMenu)
  tray.on('double-click', () => { if (win) { win.show(); win.focus() } })
  tray.on('click', () => { if (win) { win.show(); win.focus() } })
}

// 最小化时隐藏到托盘（而非任务栏）
function hideToTray() {
  if (win) { win.hide(); }
}

app.whenReady().then(() => {
  createWindow()
  createTray()
  log('应用启动 v' + (app.getVersion() || '1.6.2'))
  // 拦截最小化 -> 隐藏到托盘
  if (win) {
    win.on('minimize', (e) => { e.preventDefault(); hideToTray() })
    win.on('focus', () => { win.webContents.send('win-focus') })
    win.on('restore', () => { win.webContents.send('win-focus') })
    win.on('show', () => { setTimeout(() => { if (win) win.webContents.send('win-focus') }, 50) })
  }
  registerShortcuts()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('will-quit', () => { globalShortcut.unregisterAll() })

// ===== IPC =====
ipcMain.on('update-shortcuts', (_, config) => {
  shortcutConfig = Object.assign(shortcutConfig, config)
  registerShortcuts()
  if (win) win.webContents.send('shortcut-config', shortcutConfig)
})

ipcMain.on('set-mouse-ignore', (_, ignore) => {
  if (!win) return
  if (ignore) win.setIgnoreMouseEvents(true, { forward: true })
  else win.setIgnoreMouseEvents(false)
})

// ===== Statshark 玩家查询：点击🔍直接在独立窗口中打开网站 =====
ipcMain.on('open-player-search', () => {
  let searchWin = new BrowserWindow({
    width: 1100, height: 750,
    title: 'Statshark — War Thunder 玩家数据',
    webPreferences: { contextIsolation: true, nodeIntegration: false },
    autoHideMenuBar: true
  })
  searchWin.loadURL('https://statshark.net/')
  searchWin.on('closed', () => { searchWin = null })
})

ipcMain.on('set-always-on-top-level', (_, level) => {
  if (!win) return
  win.setAlwaysOnTop(true, level || 'screen-saver')
})

ipcMain.on('window-minimize', () => { if (win) win.minimize() })
ipcMain.on('window-close', () => { if (win) win.close() })
ipcMain.on('app-quit', () => { app.quit() })
ipcMain.on('window-show', () => { if (win) win.show() })
ipcMain.on('window-hide', () => { if (win) win.hide() })
ipcMain.on('open-external', (_, url) => { require('electron').shell.openExternal(url) })
ipcMain.on('switch-display', (_, d) => {
  if (win && d) win.setBounds({ x: d.bounds.x, y: d.bounds.y, width: d.size.width, height: d.size.height })
})

// ===== 屏幕截图 IPC（用于放大镜）=====
ipcMain.handle('capture-screen', async () => {
  try {
    const primaryDisplay = screen.getPrimaryDisplay()
    const { width, height } = primaryDisplay.size
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: width, height: height }
    })
    if (sources.length === 0) return { error: 'no screen source' }
    const img = sources[0].thumbnail
    const sz = img.getSize()
    if (sz.width === 0 || sz.height === 0) return { error: 'empty thumbnail' }
    return { dataUrl: img.toDataURL(), width: sz.width, height: sz.height }
  } catch (err) {
    return { error: err.message }
  }
})

// 截取指定区域（用于 AI 识别时显示缩略图）
ipcMain.handle('capture-region', async (_, rx, ry, rw, rh) => {
  try {
    const primaryDisplay = screen.getPrimaryDisplay()
    const scale = primaryDisplay.scaleFactor || 1
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: primaryDisplay.size.width, height: primaryDisplay.size.height }
    })
    if (sources.length === 0) return { error: 'no screen source' }
    const src = sources[0].thumbnail
    const crop = src.crop({ x: Math.round(rx * scale), y: Math.round(ry * scale), width: Math.round(rw * scale), height: Math.round(rh * scale) })
    return { dataUrl: crop.toDataURL(), width: rw, height: rh }
  } catch (err) {
    return { error: err.message }
  }
})

// ===== AI 图像识别测距 IPC =====
ipcMain.handle('ai-detect', async (_, coefA, coefB, unit) => {
  return new Promise((resolve) => {
    const pyScript = getPyPath('ai_detect.py')
    const args = [pyScript, String(coefA), String(coefB), unit]
    const py = spawn('python', args)
    let stdout = '', stderr = ''
    py.stdout.on('data', (d) => { stdout += d.toString() })
    py.stderr.on('data', (d) => { stderr += d.toString() })
    py.on('close', (code) => {
      if (code !== 0) { resolve({ error: 'Python error: ' + stderr }); return }
      try { resolve(JSON.parse(stdout.trim())) }
      catch (e) { resolve({ error: 'Parse error: ' + stdout }) }
    })
    py.on('error', (err) => { resolve({ error: 'Failed: ' + err.message }) })
  })
})

// ===== 实时双图模板匹配测距 =====
let realtimeProcess = null

ipcMain.handle('start-realtime', (_, imgA, imgB, interval) => {
  return new Promise((resolve) => {
    if (realtimeProcess) {
      realtimeProcess.kill()
      realtimeProcess = null
    }
    const pyScript = getPyPath('realtime_match.py')
    const args = [pyScript, imgA, imgB, String(interval || 1.0)]
    const py = spawn('python', args)
    realtimeProcess = py

    let buffer = ''
    py.stdout.on('data', (data) => {
      buffer += data.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop() // keep incomplete line
      for (const line of lines) {
        if (line.trim() && win) {
          try {
            const obj = JSON.parse(line.trim())
            win.webContents.send('realtime-result', obj)
          } catch(e) {}
        }
      }
    })
    py.stderr.on('data', (d) => {
      if (win) win.webContents.send('realtime-error', d.toString())
    })
    py.on('close', (code) => {
      realtimeProcess = null
      if (win) win.webContents.send('realtime-stopped', { code })
    })
    py.on('error', (err) => {
      realtimeProcess = null
      resolve({ error: err.message })
    })
    resolve({ started: true })
  })
})

ipcMain.handle('stop-realtime', () => {
  if (realtimeProcess) {
    realtimeProcess.kill()
    realtimeProcess = null
  }
  return { stopped: true }
})

// ===== 区域实时双图识别测距 =====
let regionProcess = null
let lastRegionArgs = null   // 记录最近一次区域识别参数，训练后热重载用

// 为识别进程设置 stdout/stderr 处理（统一用于启动和热重载）
function setupRealtimeProcess(py) {
  regionProcess = py
  let buffer = ''
  py.stdout.on('data', (data) => {
    buffer += data.toString()
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) {
      if (line.trim() && win) {
        try {
          const obj = JSON.parse(line.trim())
          // YOLOv8 输出原始 logits，需要 sigmoid 归一化到 0-1
          if (obj.a && obj.a.score > 1) obj.a.score = 1 / (1 + Math.exp(-obj.a.score))
          if (obj.b && obj.b.score > 1) obj.b.score = 1 / (1 + Math.exp(-obj.b.score))
          win.webContents.send('region-realtime-result', obj)
        } catch(e) {}
      }
    }
  })
  py.stderr.on('data', (d) => {
    const s = d.toString()
    if (/net_impl_backend|setPreferableTarget|WARN|DeprecationWarning/i.test(s)) { log('识别警告(已忽略): ' + s.trim().slice(0, 120)); return }
    log('识别错误: ' + s.trim().slice(0, 200))
    if (win) win.webContents.send('region-realtime-error', s)
  })
  py.on('close', (code) => {
    regionProcess = null
    if (win) win.webContents.send('region-realtime-stopped', { code })
  })
  py.on('error', (err) => {
    regionProcess = null
  })
}

ipcMain.handle('start-region-realtime', (_, imgA, imgB, region, threshold, interval) => {
  return new Promise((resolve) => {
    stopRegionProcess()   // 先杀掉上一个识别进程（含子进程树）
    // DPI + 多显示器换算：renderer 框选用 DIP(逻辑像素)，识别引擎截图用物理像素
    // 主显示器用 dxcam(相对坐标)；副显示器用 mss(绝对屏幕坐标)
    const primary = screen.getPrimaryDisplay()
    const disp = screen.getDisplayMatching(win ? win.getBounds() : primary.bounds)
    const scale = disp.scaleFactor || 1
    const isPrimary = disp.id === primary.id
    const physRegion = isPrimary
      ? {
          x: Math.round(region.x * scale),
          y: Math.round(region.y * scale),
          w: Math.round(region.w * scale),
          h: Math.round(region.h * scale)
        }
      : {
          x: Math.round((disp.bounds.x + region.x) * scale),
          y: Math.round((disp.bounds.y + region.y) * scale),
          w: Math.round(region.w * scale),
          h: Math.round(region.h * scale)
        }
    const captureMode = isPrimary ? 0 : 1
    lastRegionArgs = [physRegion, threshold, captureMode]   // 存物理像素，训练后热重载直接复用
    log('开始AI识别测距 region=' + physRegion.x + ',' + physRegion.y + ',' + physRegion.w + 'x' + physRegion.h + ' scale=' + scale + ' mode=' + (isPrimary ? 'primary' : 'secondary'))
    const args = [
      String(physRegion.x), String(physRegion.y), String(physRegion.w), String(physRegion.h),
      String(threshold || 0.3),
      '0.20',  // 扫描间隔：每0.2秒截图一次（CPU 足够快）
      '0.35',  // 测距间隔：每0.35秒输出一次结果
      getCollectDir(),
      '0.15',  // 采集阈值（更低=更多训练样本）
      String(captureMode)
    ]
    const exePath = prepTempRuntime()
    const py = spawn(exePath, args, {
      env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE', OMP_NUM_THREADS: '4' }
    })
    setupRealtimeProcess(py)
    resolve({ started: true })
  })
})

ipcMain.handle('stop-region-realtime', () => {
  stopRegionProcess()
  return { stopped: true }
})

// 强制终止识别进程（含子进程树，Windows 下 .kill() 只杀父进程）
function stopRegionProcess() {
  if (!regionProcess) return
  const pid = regionProcess.pid
  try {
    if (process.platform === 'win32') {
      require('child_process').spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' })
    } else {
      regionProcess.kill('SIGKILL')
    }
  } catch (e) {
    try { regionProcess.kill() } catch (e2) {}
  }
  regionProcess = null
}

// ===== 设置持久化 =====
const settingsPath = path.join(app.getPath('userData'), 'pixel-ruler-settings.json')

ipcMain.handle('save-settings', (_, settings) => {
  try {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8')
    return { saved: true }
  } catch(e) { return { error: e.message } }
})

ipcMain.handle('load-settings', () => {
  try {
    if (fs.existsSync(settingsPath)) {
      return JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
    return {}
  } catch(e) { return {} }
})

// ===== 增量训练 AI（边测距边训练）=====
function pickPython() {
  // 优先内置 trainer/python.exe
  const embedded = getPyPath('trainer/python/python.exe')
  if (fs.existsSync(embedded)) return embedded
  // 回退系统 Python（Anaconda 等常见路径，含 ultralytics+torch）
  const candidates = [
    'D:/anaconda/python.exe',
    'C:/ProgramData/anaconda3/python.exe',
    'C:/Python313/python.exe',
    'C:/Python312/python.exe',
    'C:/Python311/python.exe'
  ]
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
  }
  return 'python'
}

ipcMain.handle('retrain-check', () => {
  return new Promise((resolve) => {
    // 优先使用 PyInstaller 编译的独立 exe（完全独立，无需 Python）
    const embeddedExe = getPyPath('trainer/retrain_yolo.exe')
    if (fs.existsSync(embeddedExe)) {
      const py = spawn(embeddedExe, ['--check'])
      let out = ''
      py.stdout.on('data', (d) => { out += d.toString() })
      py.on('close', () => {
        let ok = true
        try { ok = JSON.parse(out.trim()).ok } catch (e) {}
        resolve({ embedded: true, envOk: ok, interpreter: embeddedExe, mode: 'exe' })
      })
      py.on('error', () => resolve({ embedded: true, envOk: false, interpreter: embeddedExe, mode: 'exe' }))
      return
    }
    // 回退：内置 python 解释器 或 系统 python
    const embedded = fs.existsSync(getPyPath('trainer/python/python.exe'))
    const py = spawn(pickPython(), [getPyPath('retrain_yolo.py'), '--check'])
    let out = ''
    py.stdout.on('data', (d) => { out += d.toString() })
    py.on('close', () => {
      let envOk = embedded
      try { envOk = JSON.parse(out.trim()).ok || embedded } catch (e) {}
      resolve({ embedded, envOk, interpreter: pickPython() })
    })
    py.on('error', () => resolve({ embedded, envOk: embedded, interpreter: pickPython() }))
  })
})

ipcMain.handle('install-train-env', () => {
  return new Promise((resolve) => {
    const py = spawn('python', ['-m', 'pip', 'install', 'ultralytics', '--quiet'])
    let stderr = ''
    py.stderr.on('data', (d) => { stderr += d.toString() })
    py.on('close', (code) => resolve({ code, ok: code === 0, stderr: stderr.slice(-200) }))
    py.on('error', (err) => resolve({ code: -1, ok: false, error: err.message }))
  })
})

ipcMain.handle('get-sample-count', () => {
  const dir = getCollectDir()
  try {
    const n = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.jpg')).length : 0
    return { count: n, max: 1500 }
  } catch (e) { return { count: 0, max: 1500 } }
})

ipcMain.handle('read-log', () => {
  try {
    const p = logPath()
    if (!fs.existsSync(p)) return { text: '(暂无日志)' }
    const content = fs.readFileSync(p, 'utf-8')
    const lines = content.split('\n').filter(Boolean)
    return { text: lines.slice(-120).join('\n') }
  } catch (e) { return { text: '(读取失败)' } }
})

ipcMain.handle('export-log', () => {
  try {
    const src = logPath()
    if (!fs.existsSync(src)) return { ok: false, msg: '暂无日志' }
    const dst = path.join(app.getPath('desktop'), 'WT屏幕像素测距-系统日志.txt')
    fs.copyFileSync(src, dst)
    log('导出日志到桌面: ' + dst)
    return { ok: true, path: dst }
  } catch (e) { return { ok: false, msg: e.message } }
})

ipcMain.handle('filter-samples', () => {
  return new Promise((resolve) => {
    const collected = getCollectDir()
    const script = getPyPath('auto_filter_samples.py')
    // 优先使用 PyInstaller 编译的独立 exe（含 cv2，完全独立）
    const filterExe = getPyPath('trainer/auto_filter_samples.exe')
    const py = fs.existsSync(filterExe)
      ? spawn(filterExe, ['--src', collected, '--conf', '0.5'], {
          env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE' }
        })
      : spawn(pickPython(), [script, '--src', collected, '--conf', '0.5'], {
          env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE' }
        })
    let out = '', err = ''
    py.stdout.on('data', (d) => { out += d.toString() })
    py.stderr.on('data', (d) => { err += d.toString() })
    py.on('close', (code) => {
      let r = {}
      try { r = JSON.parse(out.trim().split('\n').pop()) } catch (e) {}
      log('样本筛选 kept=' + (r.kept || 0) + ' dropped=' + (r.dropped || 0))
      resolve({ code, ...r })
    })
    py.on('error', (e) => resolve({ code: -1, error: e.message }))
  })
})

// 列出收集样本（缩略图 base64）供人工挑选
ipcMain.handle('list-samples', () => {
  try {
    const dir = getCollectDir()
    if (!fs.existsSync(dir)) return { samples: [] }
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.jpg')).slice(-60)
    const samples = files.map(f => {
      const buf = fs.readFileSync(path.join(dir, f))
      return { name: f, data: 'data:image/jpeg;base64,' + buf.toString('base64') }
    })
    return { samples }
  } catch (e) { return { samples: [] } }
})

// 人工确认：把选中样本复制到 clean/ 训练集 + 持久化到「采集样本」文件夹
ipcMain.handle('confirm-samples', (_, names) => {
  try {
    const dir = getCollectDir()
    const clean = path.join(dir, 'clean')
    const persist = getPersistSamplesDir()
    fs.mkdirSync(clean, { recursive: true })
    let n = 0
    for (const f of (names || [])) {
      const src = path.join(dir, f)
      if (fs.existsSync(src)) { fs.copyFileSync(src, path.join(clean, f)); n++ }
      const txt = path.join(dir, f.replace('.jpg', '.txt'))
      if (fs.existsSync(txt)) {
        fs.copyFileSync(txt, path.join(clean, f.replace('.jpg', '.txt')))
        // 同步持久化到「采集样本」文件夹
        try { fs.copyFileSync(txt, path.join(persist, f.replace('.jpg', '.txt'))) } catch (e) {}
      }
      // 图片也同步到「采集样本」
      if (fs.existsSync(src)) { try { fs.copyFileSync(src, path.join(persist, f)) } catch (e) {} }
    }
    log('人工挑选样本加入训练: ' + n + '（已持久化到采集样本文件夹）')
    return { ok: true, count: n, persistDir: persist }
  } catch (e) { return { ok: false, msg: e.message } }
})

// 删除选中的低质量样本（移入回收站，可恢复）
ipcMain.handle('delete-samples', async (_, names) => {
  try {
    const { shell } = require('electron')
    const dir = getCollectDir()
    let n = 0
    const toTrash = []
    for (const f of (names || [])) {
      const imgPath = path.join(dir, f)
      const txtPath = path.join(dir, f.replace('.jpg', '.txt'))
      const cleanImg = path.join(dir, 'clean', f)
      const cleanTxt = path.join(dir, 'clean', f.replace('.jpg', '.txt'))
      for (const p of [imgPath, txtPath, cleanImg, cleanTxt]) {
        if (fs.existsSync(p)) toTrash.push(p)
      }
      if (fs.existsSync(imgPath)) n++
    }
    // 逐个移入回收站（shell.trashItem 是异步的）
    for (const p of toTrash) {
      try { await shell.trashItem(p) } catch (e) { try { fs.unlinkSync(p) } catch (e2) {} }
    }
    log('删除低质量样本(回收站): ' + n)
    return { ok: true, count: n }
  } catch (e) { return { ok: false, msg: e.message } }
})

// 打开「采集样本」文件夹（用户可自行查看/删除持久化样本）
ipcMain.handle('open-samples-folder', () => {
  try {
    const { shell } = require('electron')
    const dir = getPersistSamplesDir()
    shell.openPath(dir)
    return { ok: true, path: dir }
  } catch (e) { return { ok: false, msg: e.message } }
})

ipcMain.handle('retrain-ai', () => {
  return new Promise((resolve) => {
    const collectedBase = getCollectDir()
    const cleanDir = path.join(collectedBase, 'clean')
    const cleanCount = fs.existsSync(cleanDir) ? fs.readdirSync(cleanDir).filter(f => f.endsWith('.jpg')).length : 0
    const collected = cleanCount >= 5 ? cleanDir : collectedBase
    // 训练输出覆盖识别引擎实际加载的 wt_best.onnx（不是 yolo_rt/ 旧路径，否则新模型不生效）
    const outOnnx = getPyPath('wt_best.onnx')
    const script = getPyPath('retrain_yolo.py')
    const wasRunning = !!regionProcess
    // 训练期间暂停样本采集（避免训练过程中采集的数据污染训练集）
    if (regionProcess && regionProcess.stdin && regionProcess.stdin.writable) {
      try { regionProcess.stdin.write('pause\n') } catch (e) {}
    }
    // 训练：优先使用 .py + 系统 Python（已修复 yolov8n.pt 加载逻辑），exe 中包含旧代码待重建
    const trainExe = getPyPath('trainer/retrain_yolo.exe')
    const hasExe = fs.existsSync(trainExe)
    const pythonPath = pickPython()
    const canUsePython = !!pythonPath && pythonPath !== 'python'
    // 优先 Python（有修复），否则回退 exe
    const usePython = canUsePython
    log('开始训练AI mode=' + (usePython ? 'python' : 'exe'))
    const commonArgs = [
      collected, outOnnx,
      '--augment', '--pseudo_label', '--negative_samples',
      '--freeze', '10', '--lr0', '0.001', '--epochs', '50',
      '--imgsz', '640', '--batch', '4', '--device', 'cpu'
    ]
    const commonEnv = { env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE', OMP_NUM_THREADS: '4', PYTHONUNBUFFERED: '1' } }
    const py = usePython
      ? spawn(pythonPath, [script, ...commonArgs], commonEnv)
      : spawn(trainExe, commonArgs, commonEnv)
    let stdout = '', stderr = ''
    const sendProgress = (line) => {
      try {
        const o = JSON.parse(line)
        if (win) win.webContents.send('retrain-progress', o)
      } catch (e) {}
    }
    // 解析 ultralytics 训练日志：匹配 "epoch/total" 和 "X%" 进度
    let progEpoch = 0, progTotal = 0, progPct = 0
    const progMatch = (s) => /^\s*(\d+)\/(\d+)\s+.*?(\d+)%/.exec(s)
    py.stdout.on('data', (d) => {
      stdout += d.toString()
      // ultralytics 进度条用 \r 覆盖式输出，且带 ANSI 转义序列，需先清理再分割解析
      stdout.split(/[\r\n]+/).forEach(l => {
        if (!l.trim()) return
        // 去除 ANSI 转义序列（\u001b[K 清屏、颜色码等），否则正则 ^\s* 匹配不到
        const clean = l.replace(/\u001b\[[0-9;]*[a-zA-Z]/g, '')
        if (!clean.trim()) return
        // 先尝试 JSON 状态事件
        if (clean.trim().startsWith('{')) { sendProgress(clean.trim()); return }
        // 再解析进度
        const m = progMatch(clean)
        if (m) {
          progEpoch = parseInt(m[1]); progTotal = parseInt(m[2]); progPct = parseInt(m[3])
          // 计算总训练进度：已完成 epoch 占整体比例
          const totalPct = progTotal > 0 ? Math.round(((progEpoch - 1) + progPct / 100) / progTotal * 100) : progPct
          if (win) win.webContents.send('retrain-progress', { status: 'progress', epoch: progEpoch, total: progTotal, pct: totalPct })
        }
      })
    })
    py.stderr.on('data', (d) => { stderr += d.toString() })
    py.on('close', (code) => {
      let result = {}
      try { result = JSON.parse(stdout.trim().split('\n').pop()) } catch (e) {}
      log('训练结束 status=' + (result.status || 'unknown') + ' code=' + code + (result.samples ? ' samples=' + result.samples : ''))
      // 训练完成后恢复样本采集
      if (regionProcess && regionProcess.stdin && regionProcess.stdin.writable) {
        try { regionProcess.stdin.write('resume\n') } catch (e) {}
      }
      // 训练完成且之前在识别 -> 热重载新模型
      if (result.status === 'done' && wasRunning && lastRegionArgs) {
        const [region, threshold, captureMode] = lastRegionArgs
        stopRegionProcess()   // 训练后热重载前先杀旧进程
        const args = [
          String(region.x), String(region.y), String(region.w), String(region.h),
          String(threshold || 0.3), '0.20', '0.35',
          getCollectDir(),
          '0.15',   // 采集阈值（更低=更多训练样本）
          String(captureMode || 0)
        ]
        const exePath = prepTempRuntime()
        regionProcess = spawn(exePath, args, {
          env: { ...process.env, KMP_DUPLICATE_LIB_OK: 'TRUE', OMP_NUM_THREADS: '4' }
        })
        setupRealtimeProcess(regionProcess)
        if (win) win.webContents.send('retrain-progress', { status: 'reloaded' })
      }
      resolve({ code, result, stderr: stderr.slice(-300) })
    })
    py.on('error', (err) => resolve({ code: -1, error: err.message }))
  })
})

// ===== 语音文件选择（可更换自定义 AI 语音）=====
ipcMain.handle('select-voice-file', async () => {
  const { dialog } = require('electron')
  const result = await dialog.showOpenDialog(win, {
    title: '选择语音播报文件',
    filters: [{ name: 'Audio', extensions: ['mp3','wav','ogg','m4a'] }],
    properties: ['openFile']
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return { filePath: result.filePaths[0], fileName: path.basename(result.filePaths[0]) }
})

// ===== 自动更新：检查 GitHub Releases 最新版本 =====
// 用 Electron net.fetch（走 Chromium 网络栈，正确处理系统代理与证书，避免 Node https 的 schannel 吊销问题）
ipcMain.handle('check-update', async () => {
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 20000)
    let resp
    try {
      resp = await net.fetch(UPDATE_API, {
        headers: { 'User-Agent': 'WT-PixelRuler-Updater', 'Accept': 'application/vnd.github+json' },
        signal: controller.signal
      })
    } finally {
      clearTimeout(timer)
    }
    if (!resp.ok) {
      // 404 = 仓库无 Release，属正常但需提示
      if (resp.status === 404) return { hasUpdate: false, error: '仓库暂无发布版本（Release）' }
      throw new Error('HTTP ' + resp.status)
    }
    const data = await resp.json()
    const latest = String(data.tag_name || '').replace(/^v/i, '')
    const current = String(app.getVersion() || '0.0.0').replace(/^v/i, '')
    const hasUpdate = compareVersions(latest, current) > 0
    const asset = (data.assets || []).find(a => /\.exe$/i.test(a.name)) || (data.assets || [])[0]
    log('检查更新: latest=' + latest + ' current=' + current + ' hasUpdate=' + hasUpdate)
    return {
      hasUpdate,
      latest,
      current,
      changelog: data.body || '',
      downloadUrl: asset ? asset.browser_download_url : (data.html_url || ''),
      assetName: asset ? asset.name : '',
      assetSize: asset ? asset.size : 0
    }
  } catch (e) {
    log('检查更新失败: ' + e.message)
    return { hasUpdate: false, error: e.message }
  }
})

// ===== 下载更新安装包（带进度，用 net.fetch 自动跟随重定向 + 正确处理证书/代理）=====
ipcMain.handle('download-update', async (event, url) => {
  const os = require('os')
  const dest = path.join(os.tmpdir(), 'WT屏幕像素测距-Setup.exe')
  try {
    const resp = await net.fetch(url, {
      headers: { 'User-Agent': 'WT-PixelRuler-Updater' },
      redirect: 'follow'
    })
    if (!resp.ok) throw new Error('下载失败 HTTP ' + resp.status)
    const total = parseInt(resp.headers.get('content-length') || '0', 10)
    const reader = resp.body.getReader()
    const file = fs.createWriteStream(dest)
    let received = 0
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        received += value.length
        file.write(value)
        if (win) win.webContents.send('update-progress', { received, total, pct: total ? Math.round(received / total * 100) : 0 })
      }
    } finally {
      file.end()
    }
    await new Promise((res, rej) => { file.on('finish', res); file.on('error', rej) })
    log('更新包下载完成: ' + (received / 1024 / 1024).toFixed(1) + ' MB')
    return { ok: true, path: dest, size: received }
  } catch (e) {
    log('更新包下载失败: ' + e.message)
    try { fs.unlinkSync(dest) } catch (_) {}
    return { error: e.message }
  }
})

// ===== 运行安装程序 =====
ipcMain.handle('run-installer', async (_, installerPath) => {
  try {
    shell.openPath(installerPath)
    return { ok: true }
  } catch (e) {
    return { ok: false, error: e.message }
  }
})

// ===== 读取文件为 base64（供渲染进程播放本地音频）=====
ipcMain.handle('read-file-base64', (_, filePath) => {
  try {
    const buf = fs.readFileSync(filePath)
    const ext = path.extname(filePath).slice(1).toLowerCase()
    const mime = ext === 'wav' ? 'audio/wav' : ext === 'ogg' ? 'audio/ogg' : ext === 'm4a' ? 'audio/mp4' : 'audio/mpeg'
    return { base64: buf.toString('base64'), mime }
  } catch(e) { return { error: e.message } }
})

// ===== 文件完整性校验 =====
ipcMain.handle('verify-files', () => {
  const base = process.resourcesPath ? path.dirname(process.resourcesPath) : __dirname
  const checks = [
    ['主程序 PixelRuler.exe', path.join(base, 'PixelRuler.exe')],
    ['AI识别运行时 yolo_rt.exe', getPyPath('yolo_rt/yolo_rt.exe')],
    ['AI模型 wt_best.onnx', getPyPath('yolo_rt/wt_best.onnx')],
    ['语音运行时 tts_speak.exe', getPyPath('tts_speak.exe')],
    ['应用图标 icon.ico', getPyPath('icon.ico')]
  ]
  const results = checks.map(([name, p]) => ({ name, ok: fs.existsSync(p) }))
  return { results, allOk: results.every(r => r.ok) }
})

ipcMain.handle('select-image', async () => {
  const { dialog } = require('electron')
  const result = await dialog.showOpenDialog(win, {
    title: '选择参考图片',
    filters: [{ name: 'Images', extensions: ['png','jpg','jpeg','bmp'] }],
    properties: ['openFile']
  })
  if (result.canceled || result.filePaths.length === 0) return null
  const filePath = result.filePaths[0]
  const buffer = fs.readFileSync(filePath)
  const ext = path.extname(filePath).slice(1).toLowerCase()
  const mime = ext === 'png' ? 'image/png' : ext === 'bmp' ? 'image/bmp' : 'image/jpeg'
  return {
    dataUrl: 'data:' + mime + ';base64,' + buffer.toString('base64'),
    fileName: path.basename(filePath),
    filePath: filePath
  }
})

// ===== 曼波口音语音播报 IPC =====
ipcMain.handle('speak-text', async (_, text, voice, rate, pitch) => {
  return new Promise((resolve) => {
    const args = [text]
    if (voice) args.push(voice)
    if (rate) args.push(rate)
    if (pitch) args.push(pitch)
    const rt = getRuntime('tts_speak.exe', 'tts_speak.py', args)
    const py = spawn(rt.cmd, rt.args)
    let stdout = '', stderr = ''
    py.stdout.on('data', (d) => { stdout += d.toString() })
    py.stderr.on('data', (d) => { stderr += d.toString() })
    py.on('close', (code) => {
      if (code !== 0) { resolve({ error: 'TTS error: ' + stderr }); return }
      try { resolve(JSON.parse(stdout.trim())) }
      catch (e) { resolve({ error: 'Parse error: ' + stdout }) }
    })
    py.on('error', (err) => { resolve({ error: 'Failed: ' + err.message }) })
  })
})
