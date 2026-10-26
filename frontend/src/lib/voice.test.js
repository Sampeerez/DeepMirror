import { beforeAll, describe, expect, it } from 'vitest';

let DEFAULT_VOICE;
let streamSettingsPayload;
let restoreStoredVoice;
let withChanges;
let recommendedVoice;

beforeAll(async () => {
  // constants.js reads `location` at module scope.
  globalThis.location = { protocol: 'http:', host: 'localhost:8000' };
  ({
    DEFAULT_VOICE, streamSettingsPayload, restoreStoredVoice, withChanges, recommendedVoice,
  } = await import('./voice'));
});

it('builds the WebSocket settings message', () => {
  const payload = streamSettingsPayload({ ...DEFAULT_VOICE, pitch: 2 }, { gain: { enabled: true } }, true,
    { silenceSaver: false, silenceThresholdDb: -60 });
  expect(payload).toMatchObject({ pitch_shift: 2, bypass: true, silence_saver: false, silence_threshold_db: -60 });
  expect(payload.effects.gain.enabled).toBe(true);
});

describe('recommended defaults', () => {
  it('starts at pitch 0, formant 0 and automatic pitch', () => {
    expect(DEFAULT_VOICE).toMatchObject({ pitch: 0, formant: 0, autoPitch: true, customized: [] });
  });

  it('keeps only the settings the user changed', () => {
    const stored = { ...DEFAULT_VOICE, indexRate: 0.3, protect: 0.1, customized: ['indexRate'] };
    const voice = restoreStoredVoice(stored);
    expect(voice.indexRate).toBe(0.3);
    expect(voice.protect).toBe(DEFAULT_VOICE.protect);
  });

  it('records changes, and a moved pitch slider means manual pitch', () => {
    const voice = withChanges(DEFAULT_VOICE, { pitch: 5 });
    expect(voice).toMatchObject({ pitch: 5, autoPitch: false });
    expect(voice.customized.sort()).toEqual(['autoPitch', 'pitch']);
    expect(withChanges(voice, { pitch: 0, autoPitch: true }).autoPitch).toBe(true);
  });

  it('restores everything with the server-recommended detector', () => {
    expect(recommendedVoice('rmvpe')).toEqual({ ...DEFAULT_VOICE, f0Method: 'rmvpe', customized: [] });
  });

  it('sends the pitch mode', () => {
    const payload = streamSettingsPayload(DEFAULT_VOICE, {}, false, { silenceSaver: true, silenceThresholdDb: -60 });
    expect(payload.pitch_mode).toBe('auto');
  });

  it('sends the clean-up around the model with recommended defaults', () => {
    const payload = streamSettingsPayload(DEFAULT_VOICE, {}, false, { silenceSaver: true, silenceThresholdDb: -60 });
    expect(payload).toMatchObject({ isolate_voice: true, noise_reduction_db: 60, ambient_db: -60, match_loudness: true });
    const custom = withChanges(DEFAULT_VOICE, { ambientDb: -20 });
    expect(custom.customized).toContain('ambientDb');
    expect(restoreStoredVoice({ ...custom, noiseReductionDb: 3 }).noiseReductionDb).toBe(60);
  });
});
