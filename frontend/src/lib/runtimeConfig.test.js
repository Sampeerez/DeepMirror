import { describe, expect, it } from 'vitest';
import { mergeRuntimeConfig, persistedRuntimeConfig, updateRuntimeConfig } from './runtimeConfig';

const SERVER = {
  sample_rate: 48000,
  chunk_size: 8192,
  silence_saver: true,
  silence_threshold_db: -60,
  default_f0_method: 'rmvpe',
  device: { name: 'GPU', device: 'cuda:0' },
  assets: [{ id: 'hubert', state: 'ready' }],
  stream: { lookahead_seconds: 0.15 },
  f0_methods: [{ id: 'rmvpe', available: true, realtime: true, offline: true }],
};

describe('runtime config', () => {
  it('starts from the server and keeps its facts', () => {
    const config = mergeRuntimeConfig(SERVER);
    expect(config.chunkSize).toBe(8192);
    expect(config.defaultF0Method).toBe('rmvpe');
    expect(config.device.device).toBe('cuda:0');
    expect(config.f0Methods.find((m) => m.id === 'rmvpe').available).toBe(true);
  });

  it('prefers stored choices but drops stale oversized buffers', () => {
    expect(mergeRuntimeConfig(SERVER, { chunkSize: 4096 }).chunkSize).toBe(4096);
    expect(mergeRuntimeConfig(SERVER, { chunkSize: 48000 }).chunkSize).toBe(8192);
    expect(mergeRuntimeConfig(SERVER, { silenceThresholdDb: -200 }).silenceThresholdDb).toBe(-60);
  });

  it('persists only the settings the user changed', () => {
    const current = mergeRuntimeConfig(SERVER);
    const next = updateRuntimeConfig(current, { silenceThresholdDb: -55 });
    expect(persistedRuntimeConfig(null, { silenceThresholdDb: -55 }, next)).toEqual({ silenceThresholdDb: -55 });
    const later = updateRuntimeConfig(next, { chunkSize: 4096 });
    expect(persistedRuntimeConfig({ silenceThresholdDb: -55 }, { chunkSize: 4096 }, later))
      .toEqual({ silenceThresholdDb: -55, chunkSize: 4096 });
  });

  it('follows the server chunk size for hardware when the user never chose one', () => {
    expect(mergeRuntimeConfig({ ...SERVER, chunk_size: 16384 }, { silenceThresholdDb: -55 }).chunkSize).toBe(16384);
  });

  it('ignores invalid partial updates', () => {
    const current = mergeRuntimeConfig(SERVER);
    const next = updateRuntimeConfig(current, { chunkSize: -1, silenceSaver: 'yes', silenceThresholdDb: -60 });
    expect(next.chunkSize).toBe(8192);
    expect(next.silenceSaver).toBe(true);
    expect(next.silenceThresholdDb).toBe(-60);
  });
});
