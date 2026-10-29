import { describe, expect, it } from 'vitest';
import { audioDelayMs, displayDelayMs, dueFrame, settledDelay } from './avSync';

describe('audioDelayMs', () => {
  it('adds capture, round trip, server look-ahead, playback buffer and output latency', () => {
    const delay = audioDelayMs({ chunkSize: 8192, sampleRate: 48000, roundTripMs: 140, algorithmicMs: 206,
      bufferedSamples: 8192, outputLatency: 0.02 });
    expect(delay).toBe(Math.round(170.67 + 140 + 206 + 170.67 + 20));
  });

  it('assumes one buffered chunk when the playback buffer is unknown, and 0 without a stream', () => {
    expect(audioDelayMs({ chunkSize: 4800, sampleRate: 48000, roundTripMs: 0, algorithmicMs: 0 })).toBe(200);
    expect(audioDelayMs({ chunkSize: 0, sampleRate: 48000 })).toBe(0);
  });
});

describe('dueFrame', () => {
  const queue = [{ t: 0 }, { t: 33 }, { t: 66 }, { t: 100 }];

  it('shows the newest frame captured at least the delay ago and drops the older ones', () => {
    const { frame, queue: rest } = dueFrame(queue, 600, 520);
    expect(frame.t).toBe(66);
    expect(rest.map((f) => f.t)).toEqual([100]);
  });

  it('shows nothing until the first frame is due', () => {
    expect(dueFrame(queue, 400, 520).frame).toBeNull();
  });
});

it('ignores small changes of the delay', () => {
  expect(settledDelay(500, 510)).toBe(500);
  expect(settledDelay(500, 540)).toBe(540);
  expect(settledDelay(null, 300)).toBe(300);
});

it('never shows frames sooner than they come back, and otherwise follows the voice', () => {
  const latencies = Array.from({ length: 20 }, (_, i) => 150 + i * 5);   // 150..245 ms
  expect(displayDelayMs(120, latencies)).toBe(240 + 34);
  expect(displayDelayMs(650, latencies)).toBe(650);
  expect(displayDelayMs(0, [])).toBe(0);
});
