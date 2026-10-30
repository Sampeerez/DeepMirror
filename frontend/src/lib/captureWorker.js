// Camera frames to JPEG, off the page's main thread.
//
// The page hands over the camera track's frame stream (MediaStreamTrackProcessor). Each
// frame is drawn on an OffscreenCanvas and encoded here, so capture keeps the camera's own
// rate even while the page, the voice and React are busy. Only the newest frame waits:
// when an encode is still running the next frame replaces it, and frames are dropped
// while the page says too many are on their way (``pause``).

let canvas = null;
let context = null;
let quality = 0.85;
let width = 1280;
let height = 720;
let paused = false;
let reader = null;

async function encode(frame, t) {
  if (!canvas || canvas.width !== width || canvas.height !== height) {
    canvas = new OffscreenCanvas(width, height);
    context = canvas.getContext('2d');
  }
  context.drawImage(frame, 0, 0, width, height);
  frame.close();
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
  const buffer = await blob.arrayBuffer();
  self.postMessage({ t, buffer }, [buffer]);
}

async function run(readable) {
  reader = readable.getReader();
  let busy = false;
  let waiting = null;
  const next = async () => {
    while (waiting) {
      const { frame, t } = waiting;
      waiting = null;
      busy = true;
      try {
        await encode(frame, t);
      } catch {
        // A frame that cannot be encoded is skipped.
      }
      busy = false;
    }
  };
  for (;;) {
    const { value: frame, done } = await reader.read();
    if (done) break;
    // Capture time on the page's clock (a worker's own clock starts at a different origin).
    const t = performance.timeOrigin + performance.now();
    if (paused) {
      frame.close();
      continue;
    }
    if (waiting) waiting.frame.close();
    waiting = { frame, t };
    if (!busy) next();
  }
}

self.onmessage = (event) => {
  const message = event.data;
  if (message.readable) {
    ({ width, height, quality } = message);
    run(message.readable);
  } else if ('paused' in message) {
    paused = message.paused;
  } else if (message.stop) {
    reader?.cancel().catch(() => {});
    self.close();
  }
};
