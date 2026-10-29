import { useCallback, useEffect, useState } from 'react';
import { fetchCalls, setupCalls } from '../lib/api';

const BUSY = ['downloading', 'installing'];
const isBusy = (calls) => Boolean(calls && (BUSY.includes(calls.microphone?.state) || BUSY.includes(calls.camera?.state)));

/** The virtual microphone and camera for calls: their state, and one action that installs what is missing. */
export default function useCalls() {
  const [calls, setCalls] = useState(null);
  const [error, setError] = useState(null);
  const [watching, setWatching] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setCalls(await fetchCalls());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // While installing, follow it until it settles (the first seconds may still read "idle").
  useEffect(() => {
    if (!watching) return undefined;
    const since = Date.now();
    const timer = setInterval(async () => {
      try {
        const next = await fetchCalls();
        setCalls(next);
        if (!isBusy(next) && Date.now() - since > 5000) setWatching(false);
      } catch {
        // The server may be busy; the next tick tries again.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [watching]);

  const setUp = useCallback(async () => {
    setError(null);
    try {
      setCalls(await setupCalls());
      setWatching(true);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  return { calls, error, busy: watching || isBusy(calls), refresh, setUp };
}
