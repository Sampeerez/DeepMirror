// Voice Lab state: defaults, stored-state migration and the live settings message.
import { DEFAULT_F0_METHOD } from './constants';
import { STORAGE_KEYS, readStored } from './storage';

// Mirrors the server's RVC defaults (OVC_DEFAULT_*). Pitch and formant at 0
// keep the model's own voice; with autoPitch the server moves the speaker's
// register onto the voice's (voices trained in the studio know their pitch).
export const DEFAULT_VOICE = {
  pitch: 0,
  autoPitch: true,
  formant: 0,
  f0Method: DEFAULT_F0_METHOD,
  indexRate: 0.75,
  filterRadius: 3,
  rmsMixRate: 0.25,
  protect: 0.33,
  crepeHopLength: 160,
  customized: [],
};

const TUNABLE_KEYS = Object.keys(DEFAULT_VOICE).filter((key) => key !== 'customized');

// RVC parameters a short-lived 2.x build wrote back for every user. Only this
// exact, untouched combination is migrated, so deliberate tuning survives.
const LEGACY_FIDELITY_VOICE = { indexRate: 0.85, filterRadius: 5, rmsMixRate: 0.4, protect: 0.4 };
// Pitch detectors that no longer exist; such voices fall back to the default.
const REMOVED_F0_METHODS = new Set(['rmvpe-onnx', 'crepe-onnx-full', 'crepe-onnx-tiny']);

/**
 * A stored v3 voice: settings the user changed are kept, everything else
 * follows the current recommended defaults (so improved defaults reach
 * everyone who never tuned that control).
 */
export function restoreStoredVoice(stored) {
  if (!stored || typeof stored !== 'object') return { ...DEFAULT_VOICE };
  const customized = Array.isArray(stored.customized)
    ? stored.customized.filter((key) => TUNABLE_KEYS.includes(key))
    : [];
  const voice = { ...DEFAULT_VOICE, customized };
  for (const key of customized) {
    if (key in stored) voice[key] = stored[key];
  }
  if (REMOVED_F0_METHODS.has(voice.f0Method)) {
    voice.f0Method = DEFAULT_VOICE.f0Method;
    voice.customized = customized.filter((key) => key !== 'f0Method');
  }
  return voice;
}

/** Record that the user changed these settings. */
export function withChanges(voice, partial) {
  const changed = Object.keys(partial).filter((key) => TUNABLE_KEYS.includes(key));
  // Moving the pitch slider means choosing a manual shift.
  const extra = 'pitch' in partial && !('autoPitch' in partial) ? { autoPitch: false } : {};
  if ('autoPitch' in extra) changed.push('autoPitch');
  return { ...voice, ...partial, ...extra, customized: [...new Set([...(voice.customized || []), ...changed])] };
}

/** Back to the recommended settings (the server's pitch detector for this hardware). */
export function recommendedVoice(recommendedF0Method) {
  return { ...DEFAULT_VOICE, f0Method: recommendedF0Method || DEFAULT_VOICE.f0Method, customized: [] };
}

/** True when every setting is at its recommended value. */
export function isRecommendedVoice(voice) {
  return !voice.customized?.length;
}

/** A v2 voice (every field was saved): values that differ from the defaults were deliberate. */
export function migrateStoredVoice(stored) {
  if (!stored || typeof stored !== 'object') return { ...DEFAULT_VOICE };
  const voice = { ...DEFAULT_VOICE, ...stored };
  if (REMOVED_F0_METHODS.has(voice.f0Method)) voice.f0Method = DEFAULT_VOICE.f0Method;
  const legacy = Object.entries(LEGACY_FIDELITY_VOICE).every(([key, value]) => stored[key] === value);
  if (!legacy) return voice;
  return {
    ...voice,
    indexRate: DEFAULT_VOICE.indexRate,
    filterRadius: DEFAULT_VOICE.filterRadius,
    rmsMixRate: DEFAULT_VOICE.rmsMixRate,
    protect: DEFAULT_VOICE.protect,
    // That build also forced Harvest, which cannot keep up in realtime.
    f0Method: stored.f0Method === 'harvest' ? DEFAULT_VOICE.f0Method : voice.f0Method,
  };
}

/** v2 -> v3: keep deliberate values, let the rest follow the defaults. */
export function upgradeLegacyVoice(stored) {
  if (!stored || typeof stored !== 'object') return { ...DEFAULT_VOICE };
  const migrated = migrateStoredVoice(stored);
  // v2 saved the detector on every run, so it says nothing about a choice: re-adopt
  // the server's recommendation. A non-zero pitch was set by hand.
  const customized = TUNABLE_KEYS.filter((key) => !['f0Method', 'autoPitch'].includes(key)
    && key in stored && migrated[key] !== DEFAULT_VOICE[key]);
  if (customized.includes('pitch')) customized.push('autoPitch');
  return restoreStoredVoice({ ...migrated, autoPitch: !customized.includes('pitch'), customized });
}

export function loadInitialVoice() {
  const stored = readStored(STORAGE_KEYS.voice);
  if (stored) return restoreStoredVoice(stored);
  return upgradeLegacyVoice(readStored(STORAGE_KEYS.legacyVoice));
}

/** True when the user explicitly chose a pitch detector. */
export function hasStoredF0Method() {
  return Boolean(readStored(STORAGE_KEYS.voice)?.customized?.includes('f0Method'));
}

/** The full live settings message for the audio WebSocket. */
export function streamSettingsPayload(voice, effects, bypass, runtimeConfig) {
  return {
    pitch_shift: voice.pitch,
    pitch_mode: voice.autoPitch ? 'auto' : 'manual',
    formant_shift: voice.formant,
    f0_method: voice.f0Method,
    index_rate: voice.indexRate,
    filter_radius: voice.filterRadius,
    rms_mix_rate: voice.rmsMixRate,
    protect: voice.protect,
    crepe_hop_length: voice.crepeHopLength,
    effects,
    bypass,
    silence_saver: runtimeConfig.silenceSaver,
    silence_threshold_db: runtimeConfig.silenceThresholdDb,
  };
}
