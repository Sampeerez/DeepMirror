// Voice Lab state: defaults, the settings kept per voice model and the live settings message.
import { DEFAULT_F0_METHOD } from './constants';
import { defaultEffects, effectsFromPreset } from './effects';
import { STORAGE_KEYS, readStored, writeStored } from './storage';

// Mirrors the server's RVC defaults (OVC_DEFAULT_*). Pitch and formant at 0
// keep the model's own voice; with autoPitch the server moves the speaker's
// register onto the voice's (voices trained in the studio know their pitch).
export const DEFAULT_VOICE = {
  pitch: 0,
  autoPitch: true,
  // > 1 widens your melody (livelier, less monotone); voices trained here use their own range.
  expressiveness: 1.25,
  pitchCorrection: 0,   // autotune: 0 natural, 1 every note in tune
  formant: 0,
  f0Method: DEFAULT_F0_METHOD,
  indexRate: 0.75,
  filterRadius: 3,
  rmsMixRate: 0.25,
  protect: 0.33,
  crepeHopLength: 160,
  // Around the model (live): convert speech only, clean the room noise out of
  // what the model hears, keep the room audible unconverted, match your volume.
  isolateVoice: true,
  noiseReductionDb: 60,   // the maximum: all noise removed (DeepFilterNet)
  ambientDb: -60,   // off: nothing of your raw microphone is ever played
  matchLoudness: true,
  customized: [],
};

const TUNABLE_KEYS = Object.keys(DEFAULT_VOICE).filter((key) => key !== 'customized');

/** A stored voice: the settings the user changed are kept, the rest follow the current defaults. */
export function restoreStoredVoice(stored) {
  if (!stored || typeof stored !== 'object') return { ...DEFAULT_VOICE };
  const customized = Array.isArray(stored.customized)
    ? stored.customized.filter((key) => TUNABLE_KEYS.includes(key))
    : [];
  const voice = { ...DEFAULT_VOICE, customized };
  for (const key of customized) {
    if (key in stored) voice[key] = stored[key];
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

/**
 * The settings saved for ``model``. Each voice keeps its own: one voice's
 * pitch or effects never carry over to another, and a voice without saved
 * changes starts from the automatic settings.
 */
export function loadVoiceFor(model, recommendedF0Method) {
  const saved = model ? readStored(STORAGE_KEYS.voices)?.[model] : null;
  const voice = restoreStoredVoice(saved?.voice);
  if (!voice.customized.includes('f0Method') && recommendedF0Method) voice.f0Method = recommendedF0Method;
  return { voice, effects: saved?.effects ? effectsFromPreset(saved.effects) : defaultEffects() };
}

export function storeVoiceFor(model, voice, effects) {
  if (!model) return;
  const all = readStored(STORAGE_KEYS.voices);
  writeStored(STORAGE_KEYS.voices, { ...(all && typeof all === 'object' ? all : {}), [model]: { voice, effects } });
}

/** The full live settings message for the audio WebSocket. */
export function streamSettingsPayload(voice, effects, bypass, runtimeConfig) {
  return {
    pitch_shift: voice.pitch,
    pitch_mode: voice.autoPitch ? 'auto' : 'manual',
    pitch_expressiveness: voice.expressiveness,
    pitch_correction: voice.pitchCorrection,
    formant_shift: voice.formant,
    f0_method: voice.f0Method,
    index_rate: voice.indexRate,
    filter_radius: voice.filterRadius,
    rms_mix_rate: voice.rmsMixRate,
    protect: voice.protect,
    crepe_hop_length: voice.crepeHopLength,
    isolate_voice: voice.isolateVoice,
    noise_reduction_db: voice.noiseReductionDb,
    ambient_db: voice.ambientDb,
    match_loudness: voice.matchLoudness,
    effects,
    bypass,
    silence_saver: runtimeConfig.silenceSaver,
    silence_threshold_db: runtimeConfig.silenceThresholdDb,
  };
}
