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
import useFaceClone from './hooks/useFaceClone';
import useVideoPipeline from './hooks/useVideoPipeline';
import VideoControls from './components/VideoControls';
import VideoPreview from './components/VideoPreview';
import { audioDelayMs } from './lib/avSync';
import { activateModel, deactivateModel, fetchModels, getActiveModel } from './lib/api';
import { APP_NAME } from './lib/brand';
import { defaultEffects } from './lib/effects';
import { effectsFromPresetSettings, presetSettingsFromVoice, voiceFromPreset } from './lib/presets';
import { buildTabUrl, readTabFromSearch } from './lib/navigation';
import { roomierProfile } from './lib/performance';
import { shouldTriggerShortcut } from './lib/shortcuts';
import { readStored, writeStored } from './lib/storage';
import { applyTheme, readAppliedTheme, storeTheme } from './lib/theme';
import { loadVoiceFor, recommendedVoice, withChanges } from './lib/voice';

// Loaded the first time their tab opens.
const TrainingStudio = lazy(() => import('./components/TrainingStudio'));
const FileConverter = lazy(() => import('./components/FileConverter'));
const VoiceProfile = lazy(() => import('./components/VoiceProfile'));
const FaceLibrary = lazy(() => import('./components/FaceLibrary'));
const VideoSettings = lazy(() => import('./components/VideoSettings'));
const VoiceSettings = lazy(() => import('./components/VoiceSettings'));

const ACTIVE_MODEL_POLL_MS = 5000;
const CAPTURE_SAMPLES_KEY = 'ovc_capture_samples_v1';
// After a longer chunk is chosen, the server measures it for this long before another step.
const HEADROOM_SETTLE_MS = 5000;

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
  // Each voice model keeps its own settings and effects (loaded when it becomes active).
  const [voice, setVoice] = useState(() => recommendedVoice());
  const [effects, setEffects] = useState(defaultEffects);
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

  // The tracker the server recommends for its hardware, unless the user chose one.
  const recommendedF0Ref = useRef(null);
  const adoptRecommendedF0 = useCallback((method) => {
    recommendedF0Ref.current = method;
    setVoice((previous) => (previous.customized?.includes('f0Method') ? previous : { ...previous, f0Method: method }));
  }, []);

  const runtime = useRuntimeConfig({ wsHook, isRunning, onRecommendedF0: adoptRecommendedF0 });
  const { runtimeConfig } = runtime;

  // The face clone. Its frames are shown (and sent to the virtual camera) as late as the
  // converted voice is heard, so lips and voice stay together.
  const face = useFaceClone();
  const videoDelayMs = isRunning ? audioDelayMs({
    chunkSize: runtimeConfig.chunkSize, sampleRate: pipeline.streamInfo?.sampleRate, roundTripMs: latency,
    algorithmicMs: serverStats.algorithmicDelayMs, bufferedSamples: pipeline.playbackStats?.bufferedSamples,
    outputLatency: pipeline.streamInfo?.outputLatency,
  }) : 0;
  const videoDelayRef = useRef(0);
  useEffect(() => { videoDelayRef.current = videoDelayMs; }, [videoDelayMs]);
  const getVideoDelayMs = useCallback(() => videoDelayRef.current, []);
  const video = useVideoPipeline({ getDelayMs: getVideoDelayMs });
  useSettingsSync({ voice, effects, bypass, runtimeConfig, wsHook, activeModel });

  // A newly active voice brings its own settings; nothing carries over from the previous one.
  useEffect(() => {
    if (!activeModel) return;
    const saved = loadVoiceFor(activeModel, recommendedF0Ref.current);
    setVoice(saved.voice);
    setEffects(saved.effects);
    setActivePresetId(null);
  }, [activeModel]);

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

  // B: A/B bypass. R: start/stop a take. V: video on/off.
  const { running: videoRunning, start: startVideo, stop: stopVideo } = video;
  const videoReady = face.ready && Boolean(face.activeFace);
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (isSettingsOpen) return;
      if (shouldTriggerShortcut(event, 'b')) {
        event.preventDefault();
        setBypass((prev) => !prev);
        return;
      }
      if (shouldTriggerShortcut(event, 'v') && (videoRunning || videoReady)) {
        event.preventDefault();
        if (videoRunning) stopVideo();
        else startVideo();
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
  }, [isSettingsOpen, isRunning, isRecording, startRecording, stopRecording, videoRunning, videoReady, startVideo,
    stopVideo]);

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

  // A computer too busy to keep up gets a longer chunk, one step at a time (each
  // step waits for the server to measure the new size before judging it).
  const { changeLiveChunkSize } = runtime;
  const { setChunkSize } = pipeline;
  const overloaded = isRunning && runtimeConfig.autoHeadroom && serverStats.overloaded;
  const lastHeadroomStepRef = useRef(0);
  useEffect(() => {
    if (!overloaded || Date.now() - lastHeadroomStepRef.current < HEADROOM_SETTLE_MS) return;
    const next = roomierProfile(runtimeConfig.chunkSize);
    if (!next) return;
    lastHeadroomStepRef.current = Date.now();
    setChunkSize(next.chunkSize);
    changeLiveChunkSize(next.chunkSize);
  }, [overloaded, serverStats, runtimeConfig.chunkSize, setChunkSize, changeLiveChunkSize]);

  // No stream, no latency to show.
  useEffect(() => {
    if (!isRunning) resetLatency();
  }, [isRunning, resetLatency]);

  // The voices in Models, for the Live picker (read again when Live is shown or the voice changes).
  const [voiceNames, setVoiceNames] = useState([]);
  useEffect(() => {
    if (tab !== 'studio') return undefined;
    let cancelled = false;
    fetchModels()
      .then((list) => {
        if (!cancelled) setVoiceNames((Array.isArray(list) ? list : []).filter((m) => m?.type !== 'index').map((m) => m.name));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [tab, activeModel]);

  const changeVoice = useCallback(async (name) => {
    try {
      if (!name) {
        await deactivateModel();
        setActiveModel(null);
        return;
      }
      setLoadingModel(name);
      await activateModel(name);
      setActiveModel(name);
    } catch {
      // The status line and Voices › Models show what went wrong.
    } finally {
      setLoadingModel(null);
    }
  }, []);

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

  // Back to automatic: this voice's own changes and effects are dropped.
  const handleRestoreRecommended = useCallback(() => {
    setActivePresetId(null);
    setVoice(recommendedVoice(runtimeConfig.defaultF0Method));
    setEffects(defaultEffects());
  }, [runtimeConfig.defaultF0Method]);

  // Accepts a rack or an updater over the previous rack (see EffectsRack).
  const handleEffectsChange = useCallback((nextEffects) => {
    setActivePresetId(null);
    setEffects(nextEffects);
  }, []);

  // Pressing the active preset again turns it off: the sound goes back to what it was before.
  const presetSnapshotRef = useRef(null);
  const applyPreset = useCallback((preset) => {
    if (preset.id === activePresetId && presetSnapshotRef.current) {
      const { voice: before, effects: beforeEffects } = presetSnapshotRef.current;
      presetSnapshotRef.current = null;
      setVoice(before);
      setEffects(beforeEffects);
      setActivePresetId(null);
      return;
    }
    if (!activePresetId) presetSnapshotRef.current = { voice, effects };
    // Advanced RVC values change only when the preset has them.
    setVoice((prev) => {
      const next = voiceFromPreset(preset, prev);
      const changed = Object.fromEntries(Object.entries(next).filter(([key, value]) => prev[key] !== value));
      return withChanges(prev, changed);
    });
    setEffects(effectsFromPresetSettings(preset));
    setActivePresetId(preset.id);
  }, [activePresetId, voice, effects]);

  const getCurrentSettings = useCallback(
    () => presetSettingsFromVoice(voice, effects),
    [voice, effects]
  );

  const [settingsSection, setSettingsSection] = useState('general');
  const openSettings = useCallback((section = 'general') => {
    setSettingsSection(typeof section === 'string' ? section : 'general');
    setIsSettingsOpen(true);
  }, []);
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
        {/* The two sides, each with its colour, on the same columns as the cards below. */}
        <div className="grid shrink-0 grid-cols-1 gap-4 lg:grid-cols-3">
          <AudioControls
            devices={devices}
            pipeline={pipeline}
            wsStatus={wsStatus}
            activeModel={activeModel}
            bypass={bypass}
            onBypassChange={setBypass}
            onRetryConnection={retryNow}
            voices={voiceNames}
            loadingModel={loadingModel}
            onVoiceChange={changeVoice}
          />
          <VideoControls video={video} devices={devices} face={face}
            onSetUp={() => handleTabChange('faces')} onFaceChange={face.activate} />
        </div>

        <div className="page-grid">
          <MonitorDisplay
            meters={pipeline.meters}
            latency={latency}
            latencyHistory={latencyHistory}
            serverMs={serverMs}
            serverStats={serverStats}
            bypass={bypass}
            videoPreview={video.running ? <VideoPreview canvasRef={video.setPreview} fps={video.shownFps}
              faceName={face.activeFace?.name} /> : null}
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
          <VoiceLab
            voice={voice}
            onChange={handleVoiceChange}
            autoPitchAvailable={serverStats.autoPitchAvailable}
            autoPitchShift={serverStats.autoPitchShift}
            f0Pending={serverStats.f0Pending}
            speakerProfile={serverStats.speakerProfile}
            expressivenessApplied={serverStats.expressiveness}
            onRecalibrate={() => wsHook.sendSettings({ recalibrate: true })}
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

      <div {...panelProps('voice')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('voice') ? (
          <Suspense fallback={null}>
            <VoiceProfile active={tab === 'voice'} activeModel={activeModel} inputDevices={devices.inputDevices} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('training')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('training') ? (
          <Suspense fallback={null}>
            <TrainingStudio active={tab === 'training'} onActiveModelChange={setActiveModel} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('voice-settings')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('voice-settings') ? (
          <Suspense fallback={null}>
            <VoiceSettings config={runtimeConfig} onConfigChange={runtime.changeSettings} voice={voice}
              onVoiceChange={handleVoiceChange} onRestoreRecommended={handleRestoreRecommended}
              running={pipeline.isRunning} latencyHistory={latencyHistory} serverMs={serverMs}
              serverStats={serverStats} streamInfo={pipeline.streamInfo} playbackStats={pipeline.playbackStats}
              transport={wsHook.transport} f0Methods={runtimeConfig.f0Methods}
              streamSampleRate={pipeline.streamInfo?.sampleRate} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('faces')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('faces') ? (
          <Suspense fallback={null}>
            <FaceLibrary face={face} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('face-settings')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('face-settings') ? (
          <Suspense fallback={null}>
            <VideoSettings face={face} video={video} delayMs={videoDelayMs} />
          </Suspense>
        ) : null}
      </div>

      <div {...panelProps('converter')} className="flex min-h-0 flex-1 flex-col animate-fade-in-up">
        {visited.has('converter') ? (
          <Suspense fallback={null}>
            <FileConverter voice={voice} effects={effects} activeModel={activeModel} f0Methods={runtimeConfig.f0Methods}
              active={tab === 'converter'} face={face} />
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
          <div className="relative w-full max-w-5xl">
            <GlobalSettings
              runtimeChecking={runtime.checking}
              runtimeError={runtime.error}
              onRefreshRuntime={runtime.refresh}
              config={runtimeConfig}
              onClose={closeSettings}
              theme={theme}
              onThemeChange={handleThemeChange}
              captureSamples={captureSamples}
              onCaptureSamplesChange={changeCaptureSamples}
              initialSection={settingsSection}
            />
          </div>
        </div>
      ) : null}
    </Layout>
  );
}
