// localStorage helpers that never throw (private mode, quota, disabled storage).

export const STORAGE_KEYS = {
  globalSettings: 'ovc_global_settings',
  voice: 'ovc_voice_v2',
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
