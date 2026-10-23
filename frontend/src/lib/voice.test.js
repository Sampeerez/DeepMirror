import { beforeAll, describe, expect, it } from 'vitest';

let DEFAULT_VOICE;
let migrateStoredVoice;
let streamSettingsPayload;

beforeAll(async () => {
  // constants.js reads `location` at module scope.
  globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
  ({ DEFAULT_VOICE, migrateStoredVoice, streamSettingsPayload } = await import('./voice'));
});

describe('stored voice migration', () => {
  it('fills in defaults and keeps deliberate tuning', () => {
    const voice = migrateStoredVoice({ pitch: 4, indexRate: 0.5 });
    expect(voice).toEqual({ ...DEFAULT_VOICE, pitch: 4, indexRate: 0.5 });
  });

  it('resets the untouched 2.x fidelity defaults and their forced Harvest', () => {
    const voice = migrateStoredVoice({ indexRate: 0.85, filterRadius: 5, rmsMixRate: 0.4, protect: 0.4, f0Method: 'harvest' });
    expect(voice.indexRate).toBe(DEFAULT_VOICE.indexRate);
    expect(voice.f0Method).toBe(DEFAULT_VOICE.f0Method);
  });

  it('replaces pitch detectors that no longer exist', () => {
    expect(migrateStoredVoice({ f0Method: 'rmvpe-onnx' }).f0Method).toBe(DEFAULT_VOICE.f0Method);
  });
});

it('builds the WebSocket settings message', () => {
  const payload = streamSettingsPayload({ ...DEFAULT_VOICE, pitch: 2 }, { gain: { enabled: true } }, true,
    { silenceSaver: false, silenceThresholdDb: -60 });
  expect(payload).toMatchObject({ pitch_shift: 2, bypass: true, silence_saver: false, silence_threshold_db: -60 });
  expect(payload.effects.gain.enabled).toBe(true);
});
