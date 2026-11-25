import { useEffect, useRef, useState } from 'react'
import { cameraApi } from '@/api/camera'
import { useCameraStore } from '@/stores/cameraStore'
import { IconPause, IconPlay, IconUpload, IconWebcam } from '@/components/ui/icons'
import { UploadOverlay, type UploadPhase } from '@/components/upload/UploadOverlay'
import type { CameraActionResponse } from '@/types/camera'

function applyAction(data: CameraActionResponse) {
  const running = data.status === 'started' || data.status === 'already_running' || data.status === 'streaming'
  useCameraStore.getState().setStatus({
    running,
    source: data.source ?? useCameraStore.getState().source,
    isFile: data.is_file ?? useCameraStore.getState().isFile,
    paused: false,
  })
}

export function CameraControls() {
  const running = useCameraStore(s => s.running)
  const paused = useCameraStore(s => s.paused)
  const isFile = useCameraStore(s => s.isFile)
  const cameras = useCameraStore(s => s.cameras)
  const currentCamera = useCameraStore(s => s.currentCamera)

  const [busy, setBusy] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [uploadState, setUploadState] = useState<{
    filename: string
    phase: UploadPhase
    percent: number
    error: string | null
  } | null>(null)

  // Initial load: status + camera list
  useEffect(() => {
    Promise.all([cameraApi.list().catch(() => null), cameraApi.status().catch(() => null)]).then(
      ([list, status]) => {
        if (list) useCameraStore.getState().setCameras(list.cameras, typeof list.current === 'number' ? list.current : null)
        if (status)
          useCameraStore.getState().setStatus({
            running: status.running,
            paused: status.paused,
            source: status.source,
            isFile: status.is_file,
          })
      },
    )
  }, [])

  const handleToggle = async () => {
    setBusy(true)
    try {
      const data = running ? await cameraApi.stop() : await cameraApi.start()
      if (running) {
        useCameraStore.getState().setStatus({ running: false, paused: false })
      } else {
        applyAction(data)
      }
    } catch (err) {
      console.error('Camera toggle failed:', err)
    } finally {
      setBusy(false)
    }
  }

  const handlePauseToggle = async () => {
    setBusy(true)
    try {
      const data = paused ? await cameraApi.resume() : await cameraApi.pause()
      useCameraStore.getState().setStatus({ paused: data.status === 'paused' })
    } catch (err) {
      console.error('Pause toggle failed:', err)
    } finally {
      setBusy(false)
    }
  }

  // Both buttons visible while a file is the source but the stream is stopped,
  // so the user can either switch to the webcam or upload a different file.
  const showUseCameraBtn = isFile
  const showUploadBtn = !isFile || !running

  const handleUpload = () => fileInputRef.current?.click()

  const switchToWebcam = async () => {
    setBusy(true)
    try {
      const data = await cameraApi.useWebcam()
      applyAction(data)
    } catch (err) {
      console.error('Switch to webcam failed:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusy(true)
    setUploadState({ filename: file.name, phase: 'uploading', percent: 0, error: null })
    useCameraStore.getState().setStatus({ source: 'Uploading…' })
    try {
      const data = await cameraApi.uploadVideo(file, percent => {
        setUploadState(s => (s ? { ...s, percent, phase: percent < 100 ? 'uploading' : 'processing' } : s))
      })
      setUploadState(s => (s ? { ...s, phase: 'done', percent: 100 } : s))
      applyAction(data)
      setTimeout(() => setUploadState(null), 900)
    } catch (err) {
      console.error('Upload failed:', err)
      const msg = err instanceof Error ? err.message : 'Upload failed.'
      setUploadState(s => (s ? { ...s, phase: 'error', error: msg } : s))
    } finally {
      setBusy(false)
    }
  }

  const dismissUpload = () => setUploadState(null)

  const handleCameraSwitch = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const idx = Number(e.target.value)
    setBusy(true)
    try {
      const data = await cameraApi.switchTo(idx)
      applyAction(data)
      useCameraStore.getState().setCameras(cameras, idx)
    } catch (err) {
      console.error('Camera switch failed:', err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="iris-cam-controls">
      {cameras.length > 1 && !isFile && (
        <select
          className="iris-select"
          value={currentCamera ?? 0}
          onChange={handleCameraSwitch}
          disabled={busy}
          title="Switch camera"
        >
          {cameras.map(c => (
            <option key={c.index} value={c.index}>
              {c.label}
            </option>
          ))}
        </select>
      )}

      <button
        type="button"
        onClick={handleToggle}
        disabled={busy}
        className={`iris-btn ${running ? 'iris-btn-danger' : 'iris-btn-success'}`}
        title={running ? 'Stop camera' : 'Start camera'}
      >
        <span className={`iris-btn-dot ${running ? 'iris-btn-dot-stop' : 'iris-btn-dot-start'}`} />
        {running ? 'Stop' : 'Start'}
      </button>

      {running && isFile && (
        <button
          type="button"
          onClick={handlePauseToggle}
          disabled={busy}
          className="iris-btn iris-btn-ghost"
          title={paused ? 'Resume video' : 'Pause video'}
        >
          {paused ? <IconPlay /> : <IconPause />}
          {paused ? 'Resume' : 'Pause'}
        </button>
      )}

      {showUseCameraBtn && (
        <button
          type="button"
          onClick={switchToWebcam}
          disabled={busy}
          className="iris-btn iris-btn-accent"
          title="Switch to webcam"
        >
          <IconWebcam />
          Use Camera
        </button>
      )}

      {showUploadBtn && (
        <button
          type="button"
          onClick={handleUpload}
          disabled={busy}
          className="iris-btn iris-btn-ghost"
          title="Upload a video file"
        >
          <IconUpload />
          Upload Video
        </button>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.avi,.mkv,.mov,.webm"
        hidden
        onChange={handleFile}
      />

      {uploadState && (
        <UploadOverlay
          filename={uploadState.filename}
          phase={uploadState.phase}
          percent={uploadState.percent}
          errorMessage={uploadState.error}
          onDismiss={uploadState.phase === 'error' ? dismissUpload : undefined}
        />
      )}
    </div>
  )
}
