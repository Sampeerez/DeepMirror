import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { Settings2 } from 'lucide-react';
import Layout from './components/Layout';
import StatusIndicator from './components/StatusIndicator';
import AudioControls from './components/AudioControls';
import GlobalSettings from './components/GlobalSettings';
import ModelManager from './components/ModelManager';
import MonitorDisplay from './components/MonitorDisplay';
import Visualizer from './components/Visualizer';
import VoiceLab from './components/VoiceLab';
import SoundCard from './components/SoundCard';
import Button from './components/Button';
import PresetBar from './components/PresetBar';
import Recorder from './components/Recorder';
import RuntimeReadiness from './components/RuntimeReadiness';
import usePresence from './hooks/usePresence';
import useWebSocket from './hooks/useWebSocket';
import useAudioPipeline from './hooks/useAudioPipeline';
import useAudioDevices from './hooks/useAudioDevices';
import useRuntimeConfig from './hooks/useRuntimeConfig';
import useSettingsSync from './hooks/useSettingsSync';
import { getActiveModel } from './lib/api';
import { APP_NAME } from './lib/brand';
import { defaultEffects, effectsFromPreset } from './lib/effects';
import { effectsFromPresetSettings, presetSettingsFromVoice, voiceFromPreset } from './lib/presets';
import { buildTabUrl, readTabFromSearch } from './lib/navigation';
import { shouldTriggerShortcut } from './lib/shortcuts';
import { STORAGE_KEYS, readStored, removeStored, writeStored } from './lib/storage';
import { applyTheme, readAppliedTheme, storeTheme } from './lib/theme';
import { hasStoredF0Method, loadInitialVoice, recommendedVoice, withChanges } from './lib/voice';

// Loaded the first time their tab opens.
const TrainingStudio = lazy(() => import('./components/TrainingStudio'));
const FileConverter = lazy(() => import('./components/FileConverter'));
const LogsPanel = lazy(() => import('./components/LogsPanel'));

const ACTIVE_MODEL_POLL_MS = 5000;
const CAPTURE_SAMPLES_KEY = 'ovc_capture_samples_v1';

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
  useEffect(() => { document.title = APP_NAME; }, []);
  const wsHook = useWebSocket();
  // Keeps the studio running while this page is open (it closes when the last page closes).
  usePresence();
  const pipeline = useAudioPipeline(wsHook);
  const devices = useAudioDevices();

  // Deep links: ?tab=models|converter selects a tab, ?settings opens the modal.
  const [tab, setTab] = useState(() => readTabFromSearch(window.location.search));
  // Tabs opened at least once stay mounted (and keep their state) after that.
  const [visited, setVisited] = useState(() => new Set([readTabFromSearch(window.location.search)]));
  useEffect(() => { setVisited((previous) => (previous.has(tab) ? previous : new Set([...previous, tab]))); }, [tab]);
  const [activeModel, setActiveModel] = useState(null);
  // A voice the server is still preparing (loading, warming up): never shown as active.
  const [loadingModel, setLoadingModel] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(
    () => new URLSearchParams(window.location.search).has('settings')
  );
  const [voice, setVoice] = useState(loadInitialVoice);
  const [effects, setEffects] = useState(loadInitialEffects);
  const [activePresetId, setActivePresetId] = useState(null);
  // Not persisted: bypass must never be on by surprise at the next start.
  const [bypass, setBypass] = useState(false);
  const [captureSamples, setCaptureSamples] = useState(() => readStored(CAPTURE_SAMPLES_KEY) === true);
  // index.html already applied the theme before the first paint.
  const [theme, setTheme] = useState(() => readAppliedTheme(document.documentElement));

  const { isRunning, isRecording, startRecording, stopRecording } = pipeline;
  const setPipelineCapture = pipeline.setCaptureSamples;
  useEffect(() => { setPipelineCapture?.(captureSamples); }, [setPipelineCapture, captureSamples]);
  const changeCaptureSamples = useCallback((enabled) => {
    setCaptureSamples(enabled);
    writeStored(CAPTURE_SAMPLES_KEY, enabled);
  }, []);
  const {
    status: wsStatus, latency, latencyHistory, serverMs, serverStats, retryNow,
    setOnSettingsResponse, resetLatency,
  } = wsHook;

  // First run only: use the tracker the server recommends for its hardware.
  const adoptServerF0Ref = useRef(!hasStoredF0Method());
  const adoptRecommendedF0 = useCallback((method) => {
    if (!adoptServerF0Ref.current) return;
    adoptServerF0Ref.current = false;
    setVoice((previous) => ({ ...previous, f0Method: method }));
  }, []);

  const runtime = useRuntimeConfig({ wsHook, isRunning, onRecommendedF0: adoptRecommendedF0 });
  const { runtimeConfig } = runtime;
  useSettingsSync({ voice, effects, bypass, runtimeConfig, wsHook });

  // Live audio only uses trackers that keep up (an older saved choice such as Crepe is replaced).
  useEffect(() => {
    const methods = runtimeConfig.f0Methods;
    const current = methods.find((method) => method.id === voice.f0Method);
    if (current?.available && current.realtime) return;
    const preferred = methods.find((method) => method.id === runtimeConfig.defaultF0Method && method.available)
      || methods.find((method) => method.realtime && method.available);
    if (preferred) setVoice((previous) => ({ ...previous, f0Method: preferred.id }));
  }, [runtimeConfig.f0Methods, runtimeConfig.defaultF0Method, voice.f0Method]);

  // ?tab= keeps tabs linkable and Back/Forward-able. pushState stays outside the
  // state updater (StrictMode calls updaters twice).
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

  // B: A/B bypass. R: start/stop a take.
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

  // While streaming, status messages carry the active model; poll only when idle and visible.
  useEffect(() => {
    if (isRunning) return undefined;

    let cancelled = false;
    const checkActiveModel = async () => {
      if (document.hidden) return;
      try {
        const active = await getActiveModel({ timeoutMs: ACTIVE_MODEL_POLL_MS });
        if (!cancelled) {
          setActiveModel(active?.name || active?.model || null);
          setLoadingModel(active?.loading?.name || null);
        }
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
        setLoadingModel(msg.loading_model || null);
      }
    });
  }, [setOnSettingsResponse]);

  // Playback gaps and lost frames go to the server log, so a stutter leaves a trace.
  const underruns = pipeline.playbackStats?.underruns || 0;
  const { dropped = 0, timedOut = 0 } = wsHook.transport || {};
  const { sendSettings } = wsHook;
  useEffect(() => {
    if (!isRunning || underruns + dropped + timedOut === 0) return undefined;
    const timer = setTimeout(() => sendSettings({ client_health: { underruns, dropped, timed_out: timedOut } }), 1000);
    return () => clearTimeout(timer);
  }, [isRunning, underruns, dropped, timedOut, sendSettings]);

  // No stream, no latency to show.
  useEffect(() => {
    if (!isRunning) resetLatency();
  }, [isRunning, resetLatency]);

  // --- Handlers --------------------------------------------------------------


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
    // Advanced RVC values change only when the preset has them.
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
      <Button icon={Settings2} onClick={openSettings} title="Settings and diagnostics" aria-label="Open settings" />
    </>
  );

  // Panels stay mounted (hidden) so switching tabs keeps their state.
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
      {/* Studio: a top bar, three cards of one fixed height (content scrolls
          inside a card, never the page), and the presets row. On narrow screens
          the cards stack at their natural height. */}
      <div {...panelProps('studio')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up gap-4">
        <AudioControls
          devices={devices}
          pipeline={pipeline}
          wsStatus={wsStatus}
          activeModel={activeModel}
          bypass={bypass}
          onBypassChange={setBypass}
          onRetryConnection={retryNow}
        />

        <div className="grid grid-cols-1 gap-4 lg:min-h-0 lg:flex-1 lg:grid-cols-3">
          <VoiceLab
            voice={voice}
            onChange={handleVoiceChange}
            autoPitchAvailable={serverStats.autoPitchAvailable}
            autoPitchShift={serverStats.autoPitchShift}
            f0Pending={serverStats.f0Pending}
            speakerProfile={serverStats.speakerProfile}
            expressivenessApplied={serverStats.expressiveness}
            onRecalibrate={() => {
              removeStored(STORAGE_KEYS.speakerProfile);
              wsHook.sendSettings({ recalibrate: true });
            }}
            tooSlow={pipeline.isRunning && !serverStats.inferenceSleeping && serverStats.modelMs > 0
              && serverStats.modelMs > (runtimeConfig.chunkSize / runtimeConfig.sampleRate) * 1000 * 0.95}
            hasModel={Boolean(activeModel)}
            isRunning={pipeline.isRunning}
            f0Methods={runtimeConfig.f0Methods}
          />
          <SoundCard
            voice={voice}
            onChange={handleVoiceChange}
            onRestoreRecommended={handleRestoreRecommended}
            effects={effects}
            onEffectsChange={handleEffectsChange}
            denoiser={pipeline.isRunning ? serverStats.denoiser : null}
          />
          <MonitorDisplay
            meters={pipeline.meters}
            latency={latency}
            latencyHistory={latencyHistory}
            serverMs={serverMs}
            serverStats={serverStats}
            bypass={bypass}
            visualizer={(
              <Visualizer
                getAnalysers={pipeline.getAnalysers}
                isRunning={pipeline.isRunning}
                theme={theme}
                active={tab === 'studio'}
              />
            )}
            recorder={<Recorder pipeline={pipeline} />}
          />
        </div>

        <PresetBar
          refreshRevision={runtime.revision}
          activePresetId={activePresetId}
          onApplyPreset={applyPreset}
          getCurrentSettings={getCurrentSettings}
        />
      </div>

      <div {...panelProps('models')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        <ModelManager
          activeModel={activeModel}
          onActiveModelChange={setActiveModel}
          loadingModel={loadingModel}
          active={tab === 'models'}
          engine={(
            <RuntimeReadiness config={runtimeConfig} checking={runtime.checking} error={runtime.error}
              onRefresh={runtime.refresh} />
          )}
        />
      </div>

      <div {...panelProps('training')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('training') ? (
          <Suspense fallback={null}>
            <TrainingStudio active={tab === 'training'} onActiveModelChange={setActiveModel} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('converter')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('converter') ? (
          <Suspense fallback={null}>
            <FileConverter voice={voice} effects={effects} activeModel={activeModel} f0Methods={runtimeConfig.f0Methods} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('logs')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('logs') ? (
          <Suspense fallback={null}>
            <LogsPanel active={tab === 'logs'} />
          </Suspense>
        ) : null}
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
              streamInfo={pipeline.streamInfo}
              playbackStats={pipeline.playbackStats}
              transport={wsHook.transport}
              captureSamples={captureSamples}
              onCaptureSamplesChange={changeCaptureSamples}
            />
          </div>
        </div>
      ) : null}
    </Layout>
  );
}
