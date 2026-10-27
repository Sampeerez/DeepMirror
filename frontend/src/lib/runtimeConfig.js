// Server runtime configuration (/api/config) merged with the user's stored
// stream preferences.
import { FALLBACK_F0_METHODS, normalizeF0Capabilities } from './f0Methods';
import { PERFORMANCE_PROFILES } from './performance';

// Largest buffer the settings dialog offers; anything above is a stale value.
const MAX_STORED_CHUNK_SIZE = 16384;

export const DEFAULT_RUNTIME_CONFIG = {
  sampleRate: 48000,
  chunkSize: PERFORMANCE_PROFILES[0].chunkSize,
  silenceSaver: true,
  silenceThresholdDb: -60,
  // A longer chunk, automatically, when the computer cannot keep up.
  autoHeadroom: true,
  f0Methods: FALLBACK_F0_METHODS,
  defaultF0Method: null,
  device: null,
  assets: [],
  stream: null,
};

function normalizePositiveInt(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.round(numeric) : fallback;
}

function normalizeSilenceThreshold(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= -80 && numeric <= -20 ? numeric : fallback;
}

/** Server-reported facts only: capabilities, device, model assets. */
export function serverFacts(config) {
  return {
    f0Methods: normalizeF0Capabilities(config?.f0_methods),
    defaultF0Method: typeof config?.default_f0_method === 'string' ? config.default_f0_method : null,
    device: config?.device ?? null,
    assets: Array.isArray(config?.assets) ? config.assets : [],
    stream: config?.stream ?? null,
  };
}

export function mergeRuntimeConfig(config, stored = null) {
  const base = DEFAULT_RUNTIME_CONFIG;
  const serverChunk = normalizePositiveInt(config?.chunk_size, base.chunkSize);
  const storedChunk = normalizePositiveInt(stored?.chunkSize, serverChunk);
  return {
    sampleRate: normalizePositiveInt(stored?.sampleRate, normalizePositiveInt(config?.sample_rate, base.sampleRate)),
    chunkSize: storedChunk > MAX_STORED_CHUNK_SIZE ? serverChunk : storedChunk,
    silenceSaver: typeof stored?.silenceSaver === 'boolean'
      ? stored.silenceSaver
      : (typeof config?.silence_saver === 'boolean' ? config.silence_saver : base.silenceSaver),
    silenceThresholdDb: normalizeSilenceThreshold(
      stored?.silenceThresholdDb,
      normalizeSilenceThreshold(config?.silence_threshold_db, base.silenceThresholdDb)
    ),
    autoHeadroom: typeof stored?.autoHeadroom === 'boolean' ? stored.autoHeadroom : base.autoHeadroom,
    ...serverFacts(config),
  };
}

const USER_KEYS = ['sampleRate', 'chunkSize', 'silenceSaver', 'silenceThresholdDb', 'autoHeadroom'];

/**
 * The persisted slice: only the settings the user actually changed. Anything
 * never touched keeps following the server, whose defaults fit the hardware
 * (a larger chunk on CPU-only machines, for instance).
 */
export function persistedRuntimeConfig(previous, partial, next) {
  const saved = {};
  for (const key of USER_KEYS) {
    if (previous && key in previous) saved[key] = previous[key];
    if (partial && key in partial) saved[key] = next[key];
  }
  return saved;
}

/** Apply a partial settings change, keeping invalid fields at their current value. */
export function updateRuntimeConfig(current, partial) {
  return {
    ...current,
    sampleRate: normalizePositiveInt(partial.sampleRate, current.sampleRate),
    chunkSize: normalizePositiveInt(partial.chunkSize, current.chunkSize),
    silenceSaver: typeof partial.silenceSaver === 'boolean' ? partial.silenceSaver : current.silenceSaver,
    silenceThresholdDb: normalizeSilenceThreshold(partial.silenceThresholdDb, current.silenceThresholdDb),
    autoHeadroom: typeof partial.autoHeadroom === 'boolean' ? partial.autoHeadroom : current.autoHeadroom,
  };
}
