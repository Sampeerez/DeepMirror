import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { displayDelayMs, dueFrame, settledDelay } from '../lib/avSync';
import { readStored, writeStored } from '../lib/storage';

const VIDEO_WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/video`;
const CAMERA_KEY = 'ovc_camera_v1';
const WIDTH = 1280;
const HEIGHT = 720;
const FPS = 30;
const JPEG_QUALITY = 0.85;
// Frames on their way to the server and back. The server holds each one until the next face swap
// after it is done (to blend between swaps), about 0.3 s, so a second of frames may be out at once;
// more means the server is behind, and capture skips frames.
const MAX_IN_FLIGHT = 30;
const MAX_QUEUE = 90;

function frameMessage(t, jpeg) {
  const out = new Uint8Array(8 + jpeg.byteLength);
  new DataView(out.buffer).setFloat64(0, t, true);
  out.set(new Uint8Array(jpeg), 8);
  return out.buffer;
}

/**
 * The face clone in the browser: the camera goes to the server as JPEG frames
 * stamped with their capture time, and the cloned frames come back. They are
 * shown ``getDelayMs()`` after capture (the voice's own delay), so the face
 * moves its lips exactly when the converted voice is heard. The server can
 * also feed a virtual camera with the same delay.
 */
export default function useVideoPipeline({ getDelayMs }) {
  const stored = readStored(CAMERA_KEY) || {};
  const [cameraId, setCameraIdState] = useState(stored.cameraId || '');
  const [virtualCamera, setVirtualCameraState] = useState(Boolean(stored.virtualCamera));
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [shownFps, setShownFps] = useState(0);

  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const previewRef = useRef(null);
  const setPreview = useCallback((node) => { previewRef.current = node; }, []);
  const virtualCameraRef = useRef(virtualCamera);
  useEffect(() => { virtualCameraRef.current = virtualCamera; }, [virtualCamera]);
  const queueRef = useRef([]);
  const inFlightRef = useRef(0);
  const timersRef = useRef({ capture: null, draw: null, fps: null });
  const shownRef = useRef(0);
  const delayRef = useRef(null);
  const latenciesRef = useRef([]);    // ms from capture to the cloned frame's arrival, recent frames
  const [shownDelay, setShownDelay] = useState(0);
  const getDelayRef = useRef(getDelayMs);
  useEffect(() => { getDelayRef.current = getDelayMs; }, [getDelayMs]);

  const persist = useCallback((partial) => {
    writeStored(CAMERA_KEY, { ...(readStored(CAMERA_KEY) || {}), ...partial });
  }, []);

  const sendCamera = useCallback((on, delayMs) => {
    const ws = wsRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ camera: { on, size: [WIDTH, HEIGHT], delay_ms: Math.round(delayMs || 0) } }));
  }, []);

  const stop = useCallback(() => {
    const timers = timersRef.current;
    clearInterval(timers.capture);
    cancelAnimationFrame(timers.draw);
    clearInterval(timers.fps);
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    for (const item of queueRef.current) item.bitmap?.close?.();
    queueRef.current = [];
    inFlightRef.current = 0;
    delayRef.current = null;
    latenciesRef.current = [];
    setRunning(false);
    setStats(null);
    setShownFps(0);
  }, []);

  const start = useCallback(async () => {
    if (wsRef.current || starting) return;
    setStarting(true);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          ...(cameraId ? { deviceId: { exact: cameraId } } : {}),
          width: { ideal: WIDTH }, height: { ideal: HEIGHT }, frameRate: { ideal: FPS },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play();
      videoRef.current = video;
      const canvas = typeof OffscreenCanvas === 'function'
        ? new OffscreenCanvas(WIDTH, HEIGHT)
        : Object.assign(document.createElement('canvas'), { width: WIDTH, height: HEIGHT });
      captureCanvasRef.current = canvas;

      const ws = new WebSocket(VIDEO_WS_URL);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;
      await new Promise((resolve, reject) => {
        ws.onopen = resolve;
        ws.onerror = () => reject(new Error('Cannot reach the face clone on the server.'));
      });

      ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          let msg;
          try { msg = JSON.parse(event.data); } catch { return; }
          if (msg.type === 'stats') setStats(msg);
          else if (msg.type === 'error' || msg.type === 'camera_error') setError(msg.message);
          return;
        }
        inFlightRef.current = Math.max(0, inFlightRef.current - 1);
        const t = new DataView(event.data).getFloat64(0, true);
        const latencies = latenciesRef.current;
        latencies.push(performance.now() - t);
        if (latencies.length > 60) latencies.shift();
        try {
          const bitmap = await createImageBitmap(new Blob([event.data.slice(8)], { type: 'image/jpeg' }));
          const queue = queueRef.current;
          queue.push({ t, bitmap });
          queue.sort((a, b) => a.t - b.t);
          while (queue.length > MAX_QUEUE) queue.shift().bitmap.close();
        } catch {
          // A broken frame is skipped.
        }
      };
      ws.onclose = () => { if (wsRef.current === ws) stop(); };

      const context = canvas.getContext('2d');
      const timers = timersRef.current;
      timers.capture = setInterval(async () => {
        if (ws.readyState !== WebSocket.OPEN || inFlightRef.current >= MAX_IN_FLIGHT) return;
        if (ws.bufferedAmount > 2_000_000 || video.readyState < 2) return;
        const t = performance.now();
        context.drawImage(video, 0, 0, WIDTH, HEIGHT);
        const blob = canvas.convertToBlob
          ? await canvas.convertToBlob({ type: 'image/jpeg', quality: JPEG_QUALITY })
          : await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
        if (!blob || ws.readyState !== WebSocket.OPEN) return;
        inFlightRef.current += 1;
        ws.send(frameMessage(t, await blob.arrayBuffer()));
      }, 1000 / FPS);

      const draw = () => {
        const delay = displayDelayMs(getDelayRef.current?.() || 0, latenciesRef.current);
        const { frame, queue } = dueFrame(queueRef.current, performance.now(), delay);
        if (frame) {
          const preview = previewRef.current;
          const target = preview?.getContext('2d');
          if (target) {
            if (preview.width !== frame.bitmap.width) preview.width = frame.bitmap.width;
            if (preview.height !== frame.bitmap.height) preview.height = frame.bitmap.height;
            target.drawImage(frame.bitmap, 0, 0);
          }
          for (const old of queueRef.current) if (old !== frame && !queue.includes(old)) old.bitmap.close();
          frame.bitmap.close();
          queueRef.current = queue;
          shownRef.current += 1;
        }
        timers.draw = requestAnimationFrame(draw);
      };
      timers.draw = requestAnimationFrame(draw);
      timers.fps = setInterval(() => {
        setShownFps(shownRef.current);
        shownRef.current = 0;
        const next = settledDelay(delayRef.current, displayDelayMs(getDelayRef.current?.() || 0, latenciesRef.current));
        setShownDelay(next);
        if (next !== delayRef.current) {
          delayRef.current = next;
          if (virtualCameraRef.current) sendCamera(true, next);
        }
      }, 1000);
      if (virtualCameraRef.current) sendCamera(true, getDelayRef.current?.() || 0);
      setRunning(true);
    } catch (err) {
      stop();
      setError(err?.name === 'NotAllowedError' ? 'Camera permission is blocked in the browser.'
        : err?.name === 'NotFoundError' ? 'No camera found.' : (err?.message || 'Could not start the camera.'));
    } finally {
      setStarting(false);
    }
  }, [cameraId, starting, stop, sendCamera]);

  const setVirtualCamera = useCallback((on) => {
    setVirtualCameraState(on);
    persist({ virtualCamera: on });
    sendCamera(on, getDelayRef.current?.() || 0);
  }, [persist, sendCamera]);

  const setCameraId = useCallback((id) => {
    setCameraIdState(id);
    persist({ cameraId: id });
  }, [persist]);

  useEffect(() => stop, [stop]);

  return useMemo(() => ({
    start, stop, running, starting, error, stats, shownFps, shownDelay, setPreview, cameraId, setCameraId,
    virtualCamera, setVirtualCamera,
  }), [start, stop, running, starting, error, stats, shownFps, shownDelay, setPreview, cameraId, setCameraId,
    virtualCamera,
    setVirtualCamera]);
}
