// Voice Lab state: defaults, stored-state migration and the live settings message.
import { DEFAULT_F0_METHOD } from './constants';
import { STORAGE_KEYS, readStored } from './storage';

// Mirrors the server's RVC defaults (OVC_DEFAULT_*).
export const DEFAULT_VOICE = {
  pitch: 0,
  formant: 0,
  f0Method: DEFAULT_F0_METHOD,
  indexRate: 0.75,
  filterRadius: 3,
  rmsMixRate: 0.25,
  protect: 0.33,
  crepeHopLength: 160,
};

// RVC parameters a short-lived 2.x build wrote back for every user. Only this
// exact, untouched combination is migrated, so deliberate tuning survives.
const LEGACY_FIDELITY_VOICE = { indexRate: 0.85, filterRadius: 5, rmsMixRate: 0.4, protect: 0.4 };
// Pitch detectors that no longer exist; such voices fall back to the default.
const REMOVED_F0_METHODS = new Set(['rmvpe-onnx', 'crepe-onnx-full', 'crepe-onnx-tiny']);

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

export function loadInitialVoice() {
  return migrateStoredVoice(readStored(STORAGE_KEYS.voice));
}

/** True when the user (or an earlier session) already chose a pitch detector. */
export function hasStoredF0Method() {
  return typeof readStored(STORAGE_KEYS.voice)?.f0Method === 'string';
}

/** The full live settings message for the audio WebSocket. */
export function streamSettingsPayload(voice, effects, bypass, runtimeConfig) {
  return {
    pitch_shift: voice.pitch,
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
