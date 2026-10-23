import { useCallback, useEffect, useState } from 'react';
import { fetchConfig } from '../lib/api';
import { applyConfig } from '../lib/constants';
import { streamTimingForChunkSize } from '../lib/performance';
import {
  DEFAULT_RUNTIME_CONFIG,
  mergeRuntimeConfig,
  persistedRuntimeConfig,
  serverFacts,
  updateRuntimeConfig,
} from '../lib/runtimeConfig';
import { STORAGE_KEYS, readStored, writeStored } from '../lib/storage';

// While HuBERT/RMVPE download on first run, poll so progress and the pitch
// detectors that depend on them appear without a manual refresh.
const ASSET_POLL_MS = 3000;

function assetsPending(config) {
  return config.assets.some((asset) => asset.state === 'downloading' || asset.state === 'missing');
}

/**
 * Loads /api/config once at startup (then connects the audio socket), keeps
 * the server-reported capabilities fresh, and applies the user's stream
 * settings locally, in storage and on the live socket.
 */
export default function useRuntimeConfig({ wsHook, isRunning, onRecommendedF0 }) {
  const { connect, disconnect, status: wsStatus, sendSettings } = wsHook;
  const [runtimeConfig, setRuntimeConfig] = useState(DEFAULT_RUNTIME_CONFIG);
  const [refreshCount, setRefreshCount] = useState(0);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRefreshCount((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    const stored = readStored(STORAGE_KEYS.globalSettings);
    const adopt = (config) => {
      if (cancelled) return;
      const next = mergeRuntimeConfig(config, stored);
      applyConfig({ sample_rate: next.sampleRate, chunk_size: next.chunkSize });
      setRuntimeConfig(next);
      if (next.defaultF0Method) onRecommendedF0?.(next.defaultF0Method);
    };
    fetchConfig()
      .then(adopt)
      .catch(() => adopt({}))
      .finally(() => { if (!cancelled) connect(); });
    return () => {
      cancelled = true;
      disconnect();
    };
    // onRecommendedF0 must be stable (useCallback); this runs once per mount.
  }, [connect, disconnect, onRecommendedF0]);

  // Re-read capabilities on reconnect or on demand; never touch live transport settings.
  useEffect(() => {
    if (wsStatus !== 'connected' && refreshCount === 0) return undefined;
    let cancelled = false;
    setChecking(true);
    fetchConfig()
      .then((config) => {
        if (cancelled) return;
        setRuntimeConfig((current) => ({ ...current, ...serverFacts(config) }));
        setRevision((value) => value + 1);
        setError(null);
      })
      .catch(() => { if (!cancelled) setError('Runtime check failed. Reconnect to the server and retry.'); })
      .finally(() => { if (!cancelled) setChecking(false); });
    return () => { cancelled = true; };
  }, [wsStatus, refreshCount]);

  const pending = assetsPending(runtimeConfig);
  useEffect(() => {
    if (!pending) return undefined;
    const timer = setInterval(refresh, ASSET_POLL_MS);
    return () => clearInterval(timer);
  }, [pending, refresh]);

  // Side effects run here, once, not inside a state updater (React may call
  // updaters twice and expects them to be pure).
  const changeSettings = useCallback((partial) => {
    const next = updateRuntimeConfig(runtimeConfig, partial);
    setRuntimeConfig(next);
    writeStored(STORAGE_KEYS.globalSettings,
      persistedRuntimeConfig(readStored(STORAGE_KEYS.globalSettings), partial, next));
    applyConfig({ sample_rate: next.sampleRate, chunk_size: next.chunkSize });
    if (wsStatus !== 'connected') return;
    const live = { silence_saver: next.silenceSaver, silence_threshold_db: next.silenceThresholdDb };
    // Transport geometry cannot change under a live stream; it applies to the next one.
    if (!isRunning) {
      Object.assign(live, { sample_rate: next.sampleRate, chunk_size: next.chunkSize },
        streamTimingForChunkSize(next.chunkSize));
    }
    sendSettings(live);
  }, [runtimeConfig, wsStatus, isRunning, sendSettings]);

  return { runtimeConfig, refresh, checking, error, revision, changeSettings };
}
