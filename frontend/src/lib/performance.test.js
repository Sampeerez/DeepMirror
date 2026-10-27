import { describe, expect, it } from 'vitest';
import {
  bufferDurationMs,
  percentile,
  profileForChunkSize,
  recommendPerformanceProfile,
  roomierProfile,
  streamTimingForChunkSize,
} from './performance';

function recommendation(overrides = {}) {
  return recommendPerformanceProfile({
    latencyHistory: Array(12).fill(45),
    sampleRate: 48000,
    serverMs: 30,
    serverStats: { modelMs: 25, dspMs: 4, inferenceSleeping: false },
    isRunning: true,
    ...overrides,
  });
}

describe('performance profile recommendation', () => {
  it('prefers the highest-quality profile the measured load allows', () => {
    expect(recommendation().profile.id).toBe('quality');
  });

  it('moves to a longer chunk when network jitter exceeds the quality budget', () => {
    expect(recommendation({ latencyHistory: Array(12).fill(400) }).profile.id).toBe('steady-plus');
  });

  it('falls back to the roomiest profile and marks overload when nothing fits', () => {
    const result = recommendation({ serverMs: 400, serverStats: { modelMs: 395, dspMs: 5 } });
    expect(result.profile.id).toBe('light');
    expect(result.constrained).toBe(true);
  });

  it('recommends the CPU profile when only it has the budget', () => {
    const result = recommendation({ serverMs: 200, serverStats: { modelMs: 195, dspMs: 5 } });
    expect(result.profile.id).toBe('light');
    expect(result.constrained).toBe(false);
  });

  it('waits for enough live samples and ignores sleeping inference', () => {
    expect(recommendation({ latencyHistory: [30, 31] }).ready).toBe(false);
    expect(recommendation({ serverStats: { modelMs: 0, dspMs: 2, inferenceSleeping: true } }).ready).toBe(false);
    expect(recommendation({ isRunning: false }).ready).toBe(false);
  });
});

describe('performance helpers', () => {
  it('calculates buffer duration and preserves custom chunk sizes', () => {
    expect(bufferDurationMs(2048, 40000)).toBeCloseTo(51.2);
    expect(profileForChunkSize(4096).id).toBe('balanced');
    expect(profileForChunkSize(1024).id).toBe('custom');
  });

  it('maps profiles to server stream timing', () => {
    expect(streamTimingForChunkSize(8192)).toEqual({ lookback_seconds: 1, lookahead_seconds: 0.15 });
    expect(streamTimingForChunkSize(1000)).toEqual({});
  });

  it('uses a nearest-rank percentile', () => {
    expect(percentile([10, 20, 30, 40], 0.95)).toBe(40);
  });

  it('steps a busy stream to a longer chunk with the same context, up to 256 ms', () => {
    expect(roomierProfile(8192).chunkSize).toBe(10240);
    expect(roomierProfile(10240).chunkSize).toBe(12288);
    expect(roomierProfile(12288)).toBeNull();
    expect(roomierProfile(4096).chunkSize).toBe(8192);
  });
});
