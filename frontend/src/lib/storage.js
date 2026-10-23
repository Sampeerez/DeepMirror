// localStorage helpers that never throw (private mode, quota, disabled storage).

export const STORAGE_KEYS = {
  // v2 stores only explicit choices; v1 saved every field and pinned stale defaults.
  globalSettings: 'ovc_global_settings_v2',
  // v3 records which settings the user changed; untouched ones follow the recommended defaults.
  voice: 'ovc_voice_v3',
  legacyVoice: 'ovc_voice_v2',
  // The speaker's pitch profile learnt by the server (sent back on connect).
  speakerProfile: 'ovc_speaker_profile_v1',
  effects: 'ovc_effects_v2',
};

export function readStored(key) {
  try {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

export function writeStored(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is a convenience; the studio works without it.
  }
}

export function removeStored(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    // Storage is a convenience; the studio works without it.
  }
}
