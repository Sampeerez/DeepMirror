// Translation between the Voice Lab's `voice` state and a preset's `settings`
// payload. Mirrors backend/presets.py.
//
// The advanced RVC fields are optional on purpose: built-in presets describe a
// voice character and say nothing about pitch detection, and user presets saved
// before this existed have no such keys. For those, applying a preset must
// leave the current advanced values alone instead of resetting them, only the
// effects are unconditional. Built-in presets carry nothing else, so they
// never undo a voice's calibrated pitch.

import { effectsFromPreset } from './effects';
import { F0_METHOD_IDS } from './f0Methods';

// preset field -> voice field, with the range the backend clamps to.
export const ADVANCED_FIELD_MAP = [
  { preset: 'index_rate', voice: 'indexRate', min: 0, max: 1 },
  { preset: 'filter_radius', voice: 'filterRadius', min: 0, max: 7, integer: true },
  { preset: 'rms_mix_rate', voice: 'rmsMixRate', min: 0, max: 1 },
  { preset: 'protect', voice: 'protect', min: 0, max: 0.5 },
  { preset: 'crepe_hop_length', voice: 'crepeHopLength', min: 64, max: 512, integer: true },
];

const F0_METHODS = F0_METHOD_IDS;

function clampNumber(value, { min, max, integer }) {
  const number = Number(value);
  if (value === null || value === '' || typeof value === 'boolean' || !Number.isFinite(number)) {
    return null;
  }
  const clamped = Math.min(Math.max(number, min), max);
  return integer ? Math.round(clamped) : clamped;
}

function normalizeF0Method(value) {
  if (typeof value !== 'string') return null;
  const method = value.trim().toLowerCase();
  return F0_METHODS.includes(method) ? method : null;
}

/** Build the `settings` payload to save for the current studio state. */
export function presetSettingsFromVoice(voice = {}, effects = {}) {
  const settings = {
    pitch_shift: Number(voice.pitch ?? 0),
    formant_shift: Number(voice.formant ?? 0),
    effects,
  };

  const f0Method = normalizeF0Method(voice.f0Method);
  if (f0Method !== null) settings.f0_method = f0Method;

  for (const field of ADVANCED_FIELD_MAP) {
    const value = clampNumber(voice[field.voice], field);
    if (value !== null) settings[field.preset] = value;
  }

  return settings;
}

/**
 * Merge a preset onto the current voice state.
 *
 * Every field changes only when the preset carries it: built-in presets are
 * effects alone and keep the voice's calibrated pitch and formant; user
 * presets restore everything they saved.
 */
export function voiceFromPreset(preset, currentVoice = {}) {
  const settings = preset?.settings || {};
  const next = { ...currentVoice };
  if ('pitch_shift' in settings) next.pitch = Number(settings.pitch_shift) || 0;
  if ('formant_shift' in settings) next.formant = Number(settings.formant_shift) || 0;

  const f0Method = normalizeF0Method(settings.f0_method);
  if (f0Method !== null) next.f0Method = f0Method;

  for (const field of ADVANCED_FIELD_MAP) {
    if (!(field.preset in settings)) continue;
    const value = clampNumber(settings[field.preset], field);
    if (value !== null) next[field.voice] = value;
  }

  return next;
}

/** Effect rack for a preset, always a full rack, defaults filled in. */
export function effectsFromPresetSettings(preset) {
  return effectsFromPreset(preset?.settings?.effects);
}

/** Which optional fields a preset actually restores (for UI/debugging). */
