import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  activateFace, addFace, buildFaceHead, deleteFace, fetchVideo, prepareVideo, updateVideoSettings,
} from '../lib/api';

const POLL_BUSY_MS = 1500;    // while the models download
const POLL_IDLE_MS = 15000;

function downloading(status) {
  return Boolean(status?.assets?.some((asset) => ['downloading', 'downloaded', 'preparing'].includes(asset.state))
    || status?.faces?.some((face) => ['building', 'queued'].includes(face.head3d?.state))
    || status?.head3d_install?.state === 'installing');
}

/** The face clone's server state (models, faces, settings), shared by the Live screen and the Video pages. */
export default function useFaceClone() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchVideo());
      setError(null);
    } catch (err) {
      setError(err.message || 'The face clone is not available.');
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const busyDownloading = downloading(status);
  useEffect(() => {
    const timer = setInterval(refresh, busyDownloading ? POLL_BUSY_MS : POLL_IDLE_MS);
    return () => clearInterval(timer);
  }, [refresh, busyDownloading]);

  const run = useCallback(async (action) => {
    setBusy(true);
    setError(null);
    try {
      const next = await action();
      if (next && typeof next === 'object' && 'faces' in next) setStatus(next);
      else await refresh();
      return true;
    } catch (err) {
      setError(err.message || 'Something went wrong.');
      return false;
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const prepare = useCallback(() => run(prepareVideo), [run]);
  const add = useCallback((files, name) => run(() => addFace(files, name)), [run]);
  const activate = useCallback((faceId) => run(() => activateFace(faceId)), [run]);
  const remove = useCallback((faceId) => run(() => deleteFace(faceId)), [run]);
  const buildHead = useCallback((faceId, quality) => run(() => buildFaceHead(faceId, quality)), [run]);
  const updateSettings = useCallback((partial) => run(async () => {
    const { settings } = await updateVideoSettings(partial);
    setStatus((current) => (current ? { ...current, settings } : current));
    return null;
  }), [run]);

  const activeFace = status?.faces?.find((face) => face.id === status.active_face) || null;

  return useMemo(() => ({
    status, error, busy, refresh, prepare, add, activate, remove, buildHead, updateSettings, activeFace,
    ready: Boolean(status?.models_ready), downloading: busyDownloading,
  }), [status, error, busy, refresh, prepare, add, activate, remove, buildHead, updateSettings, activeFace,
    busyDownloading]);
}
