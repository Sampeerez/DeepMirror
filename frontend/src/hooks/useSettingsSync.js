import { useEffect, useRef } from 'react';
import { STORAGE_KEYS, writeStored } from '../lib/storage';
import { streamSettingsPayload } from '../lib/voice';

// Slider drags produce many changes per second; settings go out (and to
// storage) once the user pauses for this long.
const SETTINGS_DEBOUNCE_MS = 140;

/**
 * Keeps the server in step with the Voice Lab and effect rack: a debounced
 * full update while editing, the bypass toggle immediately (A/B must feel
 * instant), and everything again whenever a connection opens.
 */
export default function useSettingsSync({ voice, effects, bypass, runtimeConfig, wsHook }) {
  const { status: wsStatus, sendSettings, setOnOpen } = wsHook;
  const payloadRef = useRef(null);
  const statusRef = useRef(wsStatus);

  useEffect(() => {
    payloadRef.current = streamSettingsPayload(voice, effects, bypass, runtimeConfig);
  }, [voice, effects, bypass, runtimeConfig]);

  useEffect(() => { statusRef.current = wsStatus; }, [wsStatus]);

  useEffect(() => {
    const timer = setTimeout(() => {
      writeStored(STORAGE_KEYS.voice, voice);
      writeStored(STORAGE_KEYS.effects, effects);
      if (statusRef.current === 'connected') sendSettings(payloadRef.current);
    }, SETTINGS_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [voice, effects, sendSettings]);

  useEffect(() => {
    if (wsStatus === 'connected') sendSettings({ bypass });
  }, [bypass, wsStatus, sendSettings]);

  useEffect(() => {
    setOnOpen(() => sendSettings(payloadRef.current));
  }, [setOnOpen, sendSettings]);
}
