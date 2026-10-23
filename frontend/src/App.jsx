import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Settings2 } from 'lucide-react';
import Layout from './components/Layout';
import StatusIndicator from './components/StatusIndicator';
import AudioControls from './components/AudioControls';
import GlobalSettings from './components/GlobalSettings';
import ModelManager from './components/ModelManager';
import MonitorDisplay from './components/MonitorDisplay';
import Visualizer from './components/Visualizer';
import VoiceLab from './components/VoiceLab';
import EffectsRack from './components/EffectsRack';
import PresetBar from './components/PresetBar';
import Recorder from './components/Recorder';
import FileConverter from './components/FileConverter';
import RuntimeReadiness from './components/RuntimeReadiness';
import TrainingStudio from './components/TrainingStudio';
import useWebSocket from './hooks/useWebSocket';
import useAudioPipeline from './hooks/useAudioPipeline';
import useAudioDevices from './hooks/useAudioDevices';
import useRuntimeConfig from './hooks/useRuntimeConfig';
import useSettingsSync from './hooks/useSettingsSync';
import { getActiveModel } from './lib/api';
import { defaultEffects, effectsFromPreset } from './lib/effects';
import { effectsFromPresetSettings, presetSettingsFromVoice, voiceFromPreset } from './lib/presets';
import { buildTabUrl, readTabFromSearch } from './lib/navigation';
import { shouldTriggerShortcut } from './lib/shortcuts';
import { STORAGE_KEYS, readStored } from './lib/storage';
import { applyTheme, readAppliedTheme, storeTheme } from './lib/theme';
import { hasStoredF0Method, loadInitialVoice, recommendedVoice, withChanges } from './lib/voice';

const ACTIVE_MODEL_POLL_MS = 5000;

function loadInitialEffects() {
  const stored = readStored(STORAGE_KEYS.effects);
  return stored ? effectsFromPreset(stored) : defaultEffects();
}

/** Drop ?settings from the address bar without adding a history entry. */
function clearSettingsParam() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has('settings')) return;
  url.searchParams.delete('settings');
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
}

export default function App() {
  const wsHook = useWebSocket();
  const pipeline = useAudioPipeline(wsHook);
  const devices = useAudioDevices();

  // Deep links: ?tab=models|converter selects a tab, ?settings opens the modal.
  const [tab, setTab] = useState(() => readTabFromSearch(window.location.search));
  const [activeModel, setActiveModel] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(
    () => new URLSearchParams(window.location.search).has('settings')
  );
  const [voice, setVoice] = useState(loadInitialVoice);
  const [effects, setEffects] = useState(loadInitialEffects);
  const [activePresetId, setActivePresetId] = useState(null);
  // Full conversion bypass. Deliberately not persisted: a monitoring toggle
  // should never be silently still on the next time the studio opens.
  const [bypass, setBypass] = useState(false);
  // Seeded from the attribute the blocking script in index.html already set,
  // so React adopts the pre-paint theme instead of deciding it a second time.
  const [theme, setTheme] = useState(() => readAppliedTheme(document.documentElement));

  const { isRunning, isRecording, startRecording, stopRecording } = pipeline;
  const {
    status: wsStatus, latency, latencyHistory, serverMs, serverStats, retryNow,
    setOnSettingsResponse, resetLatency,
  } = wsHook;

  // First run only: start from the detector the server recommends for its
  // hardware (RMVPE on a GPU). An explicit earlier choice is never replaced.
  const adoptServerF0Ref = useRef(!hasStoredF0Method());
  const adoptRecommendedF0 = useCallback((method) => {
    if (!adoptServerF0Ref.current) return;
    adoptServerF0Ref.current = false;
    setVoice((previous) => ({ ...previous, f0Method: method }));
  }, []);

  const runtime = useRuntimeConfig({ wsHook, isRunning, onRecommendedF0: adoptRecommendedF0 });
  const { runtimeConfig } = runtime;
  useSettingsSync({ voice, effects, bypass, runtimeConfig, wsHook });

  // Never leave the voice on a detector this server cannot run.
  useEffect(() => {
    const methods = runtimeConfig.f0Methods;
    if (methods.find((method) => method.id === voice.f0Method)?.available) return;
    const preferred = methods.find((method) => method.id === runtimeConfig.defaultF0Method && method.available)
      || methods.find((method) => method.realtime && method.available);
    if (preferred) setVoice((previous) => ({ ...previous, f0Method: preferred.id }));
  }, [runtimeConfig.f0Methods, runtimeConfig.defaultF0Method, voice.f0Method]);

  // Tab selection is reflected in ?tab= so the current view can be linked and
  // Back/Forward moves between tabs instead of leaving the app.
  // Note: pushState stays outside the state updater — StrictMode invokes
  // updaters twice in development, which would push two history entries.
  const handleTabChange = useCallback((nextTab) => {
    if (nextTab === tab) return;
    window.history.pushState({ tab: nextTab }, '', buildTabUrl(window.location.href, nextTab));
    setTab(nextTab);
  }, [tab]);

  useEffect(() => {
    const handlePopState = () => setTab(readTabFromSearch(window.location.search));
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Studio shortcuts, all single letters that stay out of the way of typing:
  // "B" toggles the A/B bypass, "R" starts/stops a take while routing runs.
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (isSettingsOpen) return;
      if (shouldTriggerShortcut(event, 'b')) {
        event.preventDefault();
        setBypass((prev) => !prev);
        return;
      }
      if (shouldTriggerShortcut(event, 'r') && isRunning) {
        event.preventDefault();
        if (isRecording) stopRecording();
        else startRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isSettingsOpen, isRunning, isRecording, startRecording, stopRecording]);

  // SPA-safe anchor scrolling (e.g. /#effects) once the page has rendered.
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (!hash) return undefined;
    const timer = setTimeout(() => {
      document.getElementById(hash)?.scrollIntoView({ block: 'start' });
    }, 300);
    return () => clearTimeout(timer);
  }, []);

  // While audio streams, the server's status messages carry the active model,
  // so polling is only needed when the studio is idle — and never while the
  // tab is in the background.
  useEffect(() => {
    if (isRunning) return undefined;

    let cancelled = false;
    const checkActiveModel = async () => {
      if (document.hidden) return;
      try {
        const active = await getActiveModel({ timeoutMs: ACTIVE_MODEL_POLL_MS });
        if (!cancelled) setActiveModel(active?.name || active?.model || null);
      } catch {
        // Server may not be ready yet, or is busy loading a checkpoint.
      }
    };

    checkActiveModel();
    const interval = setInterval(checkActiveModel, ACTIVE_MODEL_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [isRunning]);

  useEffect(() => {
    setOnSettingsResponse?.((msg) => {
      if (msg.type === 'model_change') {
        setActiveModel(msg.model || null);
      }
      if (msg.type === 'status' && msg.active_model !== undefined) {
        setActiveModel(msg.active_model);
      }
    });
  }, [setOnSettingsResponse]);

  // A stopped stream has no round trip: clear the readout and the sparkline
  // instead of leaving the last measurement on screen.
  useEffect(() => {
    if (!isRunning) resetLatency();
  }, [isRunning, resetLatency]);

  // --- Handlers --------------------------------------------------------------

  // Theme is pure presentation: it only writes an attribute and a storage key,
  // and never touches the audio pipeline or the WebSocket, so switching it
  // mid-stream cannot interrupt playback.
  const handleThemeChange = useCallback((nextTheme) => {
    const applied = applyTheme(nextTheme, document.documentElement);
    storeTheme(applied, window.localStorage);
    setTheme(applied);
  }, []);

  const handleVoiceChange = useCallback((partial) => {
    setActivePresetId(null);
    setVoice((prev) => withChanges(prev, partial));
  }, []);

  const handleRestoreRecommended = useCallback(() => {
    setActivePresetId(null);
    setVoice(recommendedVoice(runtimeConfig.defaultF0Method));
  }, [runtimeConfig.defaultF0Method]);

  // Accepts a rack or an updater over the previous rack (see EffectsRack).
  const handleEffectsChange = useCallback((nextEffects) => {
    setActivePresetId(null);
    setEffects(nextEffects);
  }, []);

  const applyPreset = useCallback((preset) => {
    // Advanced RVC values are only overwritten when the preset carries them,
    // so built-ins and pre-existing presets leave the current tuning intact.
    setVoice((prev) => {
      const next = voiceFromPreset(preset, prev);
      const changed = Object.fromEntries(Object.entries(next).filter(([key, value]) => prev[key] !== value));
      return withChanges(prev, changed);
    });
    setEffects(effectsFromPresetSettings(preset));
    setActivePresetId(preset.id);
  }, []);

  const getCurrentSettings = useCallback(
    () => presetSettingsFromVoice(voice, effects),
    [voice, effects]
  );

  const openSettings = useCallback(() => setIsSettingsOpen(true), []);
  const closeSettings = useCallback(() => {
    setIsSettingsOpen(false);
    // Otherwise a reload would reopen the modal the user just closed.
    clearSettingsParam();
  }, []);

  useEffect(() => {
    if (!isSettingsOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        closeSettings();
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isSettingsOpen, closeSettings]);

  // --- Render ----------------------------------------------------------------

  const headerActions = (
    <>
      <button
        onClick={openSettings}
        /* frost-control owns the surface, border, radius, transition and focus
           ring — the previous bg/border utilities would have overridden it. */
        className="frost-control inline-flex h-8 w-8 items-center justify-center text-fg-muted hover:text-fg-secondary"
        title="Session settings"
        aria-label="Open settings"
      >
        <Settings2 className="h-4 w-4" aria-hidden="true" />
      </button>
    </>
  );

  // Every tab panel stays mounted and is hidden with the `hidden` attribute.
  // Unmounting them threw away a converter's chosen file, an in-progress
  // render, the selected microphone and open disclosure state on every tab
  // switch — none of which the user asked to reset.
  const panelProps = (id) => ({
    role: 'tabpanel',
    id: `panel-${id}`,
    'aria-labelledby': `tab-${id}`,
    hidden: tab !== id,
  });

  return (
    <Layout
      tab={tab}
      onTabChange={handleTabChange}
      statusSlot={(
        <StatusIndicator
          wsStatus={wsStatus}
          activeModel={activeModel}
          mode={serverStats.mode}
          bypass={bypass}
        />
      )}
      headerActions={headerActions}
    >
      <div {...panelProps('studio')} className="space-y-5 animate-fade-in-up">
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="space-y-5">
            <Visualizer
              getAnalysers={pipeline.getAnalysers}
              isRunning={pipeline.isRunning}
              theme={theme}
              active={tab === 'studio'}
            />
            <AudioControls
              devices={devices}
              pipeline={pipeline}
              wsStatus={wsStatus}
              activeModel={activeModel}
              bypass={bypass}
              onBypassChange={setBypass}
              onRetryConnection={retryNow}
            />
            <Recorder pipeline={pipeline} />
          </div>

          <div className="space-y-5">
            <MonitorDisplay
              streamInfo={pipeline.streamInfo}
              playbackStats={pipeline.playbackStats}
              transport={wsHook.transport}
              meters={pipeline.meters}
              latency={latency}
              latencyHistory={latencyHistory}
              serverMs={serverMs}
              serverStats={serverStats}
              bypass={bypass}
            />
            <VoiceLab
              voice={voice}
              onChange={handleVoiceChange}
              onRestoreRecommended={handleRestoreRecommended}
              autoPitchAvailable={serverStats.autoPitchAvailable}
              autoPitchShift={serverStats.autoPitchShift}
              hasModel={Boolean(activeModel)}
              isRunning={pipeline.isRunning}
              f0Methods={runtimeConfig.f0Methods}
            />
          </div>
        </div>

        <PresetBar
          refreshRevision={runtime.revision}
          activePresetId={activePresetId}
          onApplyPreset={applyPreset}
          getCurrentSettings={getCurrentSettings}
        />

        <EffectsRack
          effects={effects}
          formantShift={voice.formant}
          onEffectsChange={handleEffectsChange}
        />
      </div>

      <div {...panelProps('models')} className="mx-auto max-w-4xl animate-fade-in-up">
        <ModelManager
          activeModel={activeModel}
          onActiveModelChange={setActiveModel}
          active={tab === 'models'}
        />
        <RuntimeReadiness config={runtimeConfig} checking={runtime.checking} error={runtime.error}
          onRefresh={runtime.refresh} />
      </div>

      <div {...panelProps('training')} className="mx-auto max-w-5xl animate-fade-in-up">
        <TrainingStudio active={tab === 'training'} onActiveModelChange={setActiveModel} />
      </div>

      <div {...panelProps('converter')} className="mx-auto max-w-3xl animate-fade-in-up">
        <FileConverter
          voice={voice}
          effects={effects}
          activeModel={activeModel}
          f0Methods={runtimeConfig.f0Methods}
        />
      </div>

      {isSettingsOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6 sm:px-6">
          <button
            className="absolute inset-0 bg-scrim backdrop-blur-md"
            onClick={closeSettings}
            aria-label="Close settings modal"
          />
          <div className="relative max-h-[calc(100svh-2.5rem)] w-full max-w-5xl overflow-y-auto">
            <GlobalSettings
              runtimeChecking={runtime.checking}
              runtimeError={runtime.error}
              onRefreshRuntime={runtime.refresh}
              config={runtimeConfig}
              onChange={runtime.changeSettings}
              disabled={pipeline.isRunning}
              onClose={closeSettings}
              theme={theme}
              onThemeChange={handleThemeChange}
              latencyHistory={latencyHistory}
              serverMs={serverMs}
              serverStats={serverStats}
              streamSampleRate={pipeline.streamInfo?.sampleRate}
            />
          </div>
        </div>
      ) : null}
    </Layout>
  );
}
