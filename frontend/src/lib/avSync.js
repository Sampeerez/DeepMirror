// Keeping the cloned face in step with the cloned voice.
//
// The voice reaches the speaker (or the call) a fixed time after it was spoken:
// one chunk of capture, the round trip with the server's render, the server's
// own look-ahead and crossfade, the playback buffer and the device's output
// latency. Every camera frame carries its capture time, so the face is shown
// exactly that long after it was captured: lips and voice line up.

/** How long after capture the converted voice is heard, in ms (0 when unknown). */
export function audioDelayMs({ chunkSize, sampleRate, roundTripMs, algorithmicMs, bufferedSamples, outputLatency }) {
  const rate = Number(sampleRate);
  if (!(rate > 0) || !(chunkSize > 0)) return 0;
  const chunk = (chunkSize / rate) * 1000;
  const buffered = bufferedSamples > 0 ? (bufferedSamples / rate) * 1000 : chunk;
  const output = outputLatency > 0 ? outputLatency * 1000 : 0;
  return Math.round(chunk + Math.max(0, roundTripMs || 0) + Math.max(0, algorithmicMs || 0) + buffered + output);
}

/**
 * The frame to show at ``now``: the newest one captured at least ``delayMs``
 * earlier. Older frames are dropped from the queue (it is sorted by time).
 * Returns ``{ frame, queue }``; ``frame`` is null while nothing is due yet.
 */
export function dueFrame(queue, now, delayMs) {
  const due = now - Math.max(0, delayMs);
  let index = -1;
  for (let i = 0; i < queue.length; i += 1) {
    if (queue[i].t <= due) index = i;
    else break;
  }
  if (index < 0) return { frame: null, queue };
  return { frame: queue[index], queue: queue.slice(index + 1) };
}

/** A delay worth telling the server about (it moves the virtual camera): changes under 25 ms are noise. */
export function settledDelay(previous, next) {
  return previous == null || Math.abs(next - previous) >= 25 ? next : previous;
}

/**
 * The delay frames are shown with: the voice's, but never less than the time
 * the cloned frames actually take to come back (90th percentile plus a frame),
 * or they would arrive overdue and some would be skipped.
 */
export function displayDelayMs(voiceDelayMs, latencies) {
  if (!latencies?.length) return Math.max(0, voiceDelayMs || 0);
  const sorted = [...latencies].sort((a, b) => a - b);
  const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))];
  return Math.round(Math.max(voiceDelayMs || 0, p90 + 34));
}
