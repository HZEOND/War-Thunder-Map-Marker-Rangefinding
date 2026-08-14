const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  onDisplaysInfo: (cb) => ipcRenderer.on('displays-info', (_, d) => cb(d)),
  onWinFocus: (cb) => { ipcRenderer.on('win-focus', () => cb()); },
  onClearMeasurement: (cb) => ipcRenderer.on('clear-measurement', () => cb()),
  onShortcutConfig: (cb) => ipcRenderer.on('shortcut-config', (_, d) => cb(d)),
  updateShortcuts: (cfg) => ipcRenderer.send('update-shortcuts', cfg),
  setMouseIgnore: (ignore) => ipcRenderer.send('set-mouse-ignore', ignore),
  setAlwaysOnTopLevel: (level) => ipcRenderer.send('set-always-on-top-level', level),
  openExternal: (url) => ipcRenderer.send('open-external', url),
  openPlayerSearch: (playerId) => ipcRenderer.send('open-player-search', playerId),
  minimize: () => ipcRenderer.send('window-minimize'),
  close: () => ipcRenderer.send('window-close'),
  quitApp: () => ipcRenderer.send('app-quit'),
  show: () => ipcRenderer.send('window-show'),
  hide: () => ipcRenderer.send('window-hide'),
  switchDisplay: (d) => ipcRenderer.send('switch-display', d),
  captureScreen: () => ipcRenderer.invoke('capture-screen'),
  captureRegion: (x, y, w, h) => ipcRenderer.invoke('capture-region', x, y, w, h),
  aiDetect: (coefA, coefB, unit) => ipcRenderer.invoke('ai-detect', coefA, coefB, unit),
  // 实时双图模板匹配
  startRealtime: (imgA, imgB, interval) => ipcRenderer.invoke('start-realtime', imgA, imgB, interval),
  stopRealtime: () => ipcRenderer.invoke('stop-realtime'),
  onRealtimeResult: (cb) => ipcRenderer.on('realtime-result', (_, d) => cb(d)),
  onRealtimeError: (cb) => ipcRenderer.on('realtime-error', (_, d) => cb(d)),
  onRealtimeStopped: (cb) => ipcRenderer.on('realtime-stopped', (_, d) => cb(d)),
  // 区域实时双图识别
  startRegionRealtime: (imgA, imgB, region, threshold, interval) =>
    ipcRenderer.invoke('start-region-realtime', imgA, imgB, region, threshold, interval),
  stopRegionRealtime: () => ipcRenderer.invoke('stop-region-realtime'),
  onRegionRealtimeResult: (cb) => ipcRenderer.on('region-realtime-result', (_, d) => cb(d)),
  onRegionRealtimeError: (cb) => ipcRenderer.on('region-realtime-error', (_, d) => cb(d)),
  onRegionRealtimeStopped: (cb) => ipcRenderer.on('region-realtime-stopped', (_, d) => cb(d)),
  // 设置持久化
  saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
  loadSettings: () => ipcRenderer.invoke('load-settings'),
  // 图片选择
  selectImage: () => ipcRenderer.invoke('select-image'),
  // 曼波口音语音播报
  speakText: (text, voice, rate, pitch) => ipcRenderer.invoke('speak-text', text, voice, rate, pitch),
  // 自定义语音文件（可更换其他 AI 语音）
  selectVoiceFile: () => ipcRenderer.invoke('select-voice-file'),
  readFileBase64: (path) => ipcRenderer.invoke('read-file-base64', path),
  // 文件完整性校验
  verifyFiles: () => ipcRenderer.invoke('verify-files'),
  // 增量训练 AI
  retrainAI: () => ipcRenderer.invoke('retrain-ai'),
  retrainCheck: () => ipcRenderer.invoke('retrain-check'),
  installTrainEnv: () => ipcRenderer.invoke('install-train-env'),
  getSampleCount: () => ipcRenderer.invoke('get-sample-count'),
  readLog: () => ipcRenderer.invoke('read-log'),
  exportLog: () => ipcRenderer.invoke('export-log'),
  filterSamples: () => ipcRenderer.invoke('filter-samples'),
  listSamples: () => ipcRenderer.invoke('list-samples'),
  confirmSamples: (names) => ipcRenderer.invoke('confirm-samples', names),
  deleteSamples: (names) => ipcRenderer.invoke('delete-samples', names),
  openSamplesFolder: () => ipcRenderer.invoke('open-samples-folder'),
  onRetrainProgress: (cb) => ipcRenderer.on('retrain-progress', (_, d) => cb(d)),
  // 自动更新
  checkUpdate: () => ipcRenderer.invoke('check-update'),
  downloadUpdate: (url) => ipcRenderer.invoke('download-update', url),
  runInstaller: (p) => ipcRenderer.invoke('run-installer', p),
  onUpdateProgress: (cb) => ipcRenderer.on('update-progress', (_, d) => cb(d))
})
