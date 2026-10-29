import React, { Suspense, lazy, memo, useMemo, useState } from 'react';
import {
  Activity, AudioLines, Camera, ChevronRight, Mic, Moon, ScanFace, ScrollText, Settings2, Sun,
} from 'lucide-react';
import Button from './Button';
import Dialog from './Dialog';
import RuntimeReadiness from './RuntimeReadiness';
import Segmented from './Segmented';
import SettingRow, { SettingsSection } from './SettingRow';
import StreamDiagnostics from './StreamDiagnostics';
import Toggle from './Toggle';
import { MiniSlider } from './Sliders';
import { APP_NAME } from '../lib/brand';
import { groupedMethods, findF0Method } from '../lib/f0Methods';
import {
  PERFORMANCE_PROFILES, bufferDurationMs, profileForChunkSize, recommendPerformanceProfile,
} from '../lib/performance';

const LogsPanel = lazy(() => import('./LogsPanel'));

const SAMPLE_RATES = [44100, 48000];
const THEMES = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
];
const SWAPPER_NAMES = { hyperswap_1b_256: 'HyperSwap', inswapper_128: 'inswapper', ghost_1_256: 'GHOST' };
// What each face model is best at, as its button label.
const SWAPPER_CHOICES = { hyperswap_1b_256: 'Sharp', inswapper_128: 'Likeness', ghost_1_256: 'Fast' };
// The buffer as one slider, from the shortest delay to the steadiest.
const BUFFERS = PERFORMANCE_PROFILES.filter((p) => p.id !== 'light').sort((a, b) => a.chunkSize - b.chunkSize);
const AUTO_ON_OFF = [
  { value: 'auto', label: 'Auto' },
  { value: 'on', label: 'On' },
  { value: 'off', label: 'Off' },
];
const STATE_TEXT = { downloading: 'Downloading…', installing: 'Installing…' };

// Rough end-to-end latency of a profile: about three chunks (capture, playback
// priming, server render) plus the server's context delays.
function latencyMs(profile, sampleRate) {
  return Math.round(3 * bufferDurationMs(profile.chunkSize, sampleRate) + profile.lookaheadSeconds * 1000 + 56);
}

function Column({ id, title, Icon, accent, children }) {
  return (
    <section className="panel flex min-h-0 flex-col p-4" aria-labelledby={`settings-${id}`}>
      <h2 id={`settings-${id}`} className="flex h-7 shrink-0 items-center gap-2 text-sm font-semibold text-fg">
        <Icon className={`h-[18px] w-[18px] ${accent}`} aria-hidden="true" />
        {title}
      </h2>
      <div className="mt-3 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">{children}</div>
    </section>
  );
}

/** A section folded by default: what most people never need to touch. */
function More({ title, children }) {
  return (
    <details className="group rounded-lg border border-line bg-raised">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.12em] text-fg-subtle [&::-webkit-details-marker]:hidden">
        <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" aria-hidden="true" />
        {title}
      </summary>
      <div className="divide-y divide-line border-t border-line">{children}</div>
    </details>
  );
}

function Stat({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-4 py-2 text-xs">
      <dt className="text-fg-subtle">{label}</dt>
      <dd className="font-mono tabular-nums text-fg-secondary">{value}</dd>
    </div>
  );
}

function CallDevice({ Icon, label, item }) {
  const state = STATE_TEXT[item?.state];
  const ok = item?.installed;
  return (
    <div className="flex items-center gap-3 px-4 py-2.5">
      <Icon className="h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />
      <span className="min-w-0 flex-1 truncate text-sm text-fg-secondary">{label}</span>
      <span className={`shrink-0 text-xs font-medium ${state ? 'text-fg-subtle' : ok ? 'text-ok-fg' : 'text-warn-fg'}`}>
        {state || (item == null ? '…' : ok ? 'Ready' : item.supported ? 'Not installed' : 'Not available')}
      </span>
    </div>
  );
}

/**
 * Settings: one page, three columns. The voice, the face, and the rest
 * (appearance, calls, engine, log). What only matters when something goes
 * wrong is folded away.
 */
function Settings({
  config, onConfigChange, voice, onVoiceChange, running, latencyHistory = [], latency = 0, serverMs = 0,
  serverStats = {}, streamInfo = null, playbackStats = null, transport = {}, f0Methods = [], face, video,
  delayMs = 0, theme, onThemeChange, captureSamples = false, onCaptureSamplesChange, runtime, calls,
}) {
  const [logsOpen, setLogsOpen] = useState(false);
  const sampleRate = streamInfo?.sampleRate || config.sampleRate;
  const profile = profileForChunkSize(config.chunkSize);
  const recommendation = useMemo(() => recommendPerformanceProfile({
    latencyHistory, sampleRate, serverMs, serverStats, isRunning: running,
  }), [latencyHistory, sampleRate, serverMs, serverStats, running]);
  const tracker = findF0Method(f0Methods, voice.f0Method);
  const liveGroups = groupedMethods(f0Methods, 'realtime');

  const faceSettings = face.status?.settings || {};
  const swapMs = face.status?.swap_ms || {};
  const stats = video.stats;
  const changeFace = (partial) => face.updateSettings(partial);
  const ms = (value) => (value == null ? '—' : `${Math.round(value)} ms`);

  const missing = calls.calls && ['microphone', 'camera'].some((key) => {
    const item = calls.calls[key];
    return item?.supported && !item.installed;
  });
  const helpUrl = calls.calls?.camera?.help_url;
  const callMessage = calls.error || calls.calls?.microphone?.message || calls.calls?.camera?.message;
  const callError = calls.error || calls.calls?.microphone?.state === 'error' || calls.calls?.camera?.state === 'error';

  return (
    <div className="page-grid">
      <Column id="voice" title="Voice" Icon={AudioLines} accent="text-ok-fg">
        <SettingsSection title="Delay">
          <SettingRow label="Buffer" stacked
            value={`${profile.label} · ${Math.round(bufferDurationMs(config.chunkSize, sampleRate))} ms`}>
            <input type="range" aria-label="Buffer" min={0} max={BUFFERS.length - 1} step={1}
              value={Math.max(0, BUFFERS.findIndex((p) => p.id === profile.id))}
              aria-valuetext={`${profile.label}, about ${latencyMs(profile, sampleRate)} milliseconds end to end`}
              onChange={(event) => onConfigChange({ chunkSize: BUFFERS[Number(event.target.value)].chunkSize })}
              className="fx-slider w-full" />
            <div className="mt-1 flex justify-between text-[10px] font-medium text-fg-faint">
              <span>Faster</span><span>Steadier</span>
            </div>
          </SettingRow>
          {recommendation.ready && recommendation.profile.chunkSize !== config.chunkSize ? (
            <SettingRow label={`${recommendation.profile.label} fits better here`}>
              <Button icon={Activity} onClick={() => onConfigChange({ chunkSize: recommendation.profile.chunkSize })}>
                Use it
              </Button>
            </SettingRow>
          ) : null}
          <SettingRow label="Sample rate">
            <Segmented label="Sample rate" hideLabel accent="voice" size="sm" value={config.sampleRate}
              onChange={(rate) => onConfigChange({ sampleRate: rate })}
              options={SAMPLE_RATES.map((rate) => ({ value: rate, label: `${rate / 1000} kHz` }))} className="w-44" />
          </SettingRow>
          <SettingRow label="Pitch tracker" htmlFor="voice-f0-method"
            hint={tracker.available ? null : tracker.reason}>
            <select id="voice-f0-method" value={tracker.id} onChange={(event) => onVoiceChange({ f0Method: event.target.value })}
              className="native-select-safe field w-44">
              {liveGroups.map((group) => (
                <optgroup key={group.section} label={group.section}>
                  {group.methods.map((method) => (
                    <option key={method.id} value={method.id} disabled={!method.available}>
                      {method.label}{method.available ? '' : ' (unavailable)'}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </SettingRow>
          {running ? <p className="px-4 py-2 text-xs text-fg-subtle">Applies the next time you press Start.</p> : null}
        </SettingsSection>

        <SettingsSection title="Silence">
          <SettingRow label="Sleep when quiet">
            <Toggle checked={config.silenceSaver} onChange={(silenceSaver) => onConfigChange({ silenceSaver })}
              label="Sleep when quiet" />
          </SettingRow>
          {config.silenceSaver ? (
            <SettingRow label="Quiet below" stacked value={`${config.silenceThresholdDb} dB`}>
              <input type="range" aria-label="Quiet below" min="-80" max="-20" step="1" value={config.silenceThresholdDb}
                aria-valuetext={`${config.silenceThresholdDb} decibels`}
                onChange={(event) => onConfigChange({ silenceThresholdDb: Number(event.target.value) })}
                className="fx-slider w-full" />
            </SettingRow>
          ) : null}
        </SettingsSection>

        <More title="Advanced">
          <SettingRow label="Longer buffer when busy">
            <Toggle checked={config.autoHeadroom} onChange={(autoHeadroom) => onConfigChange({ autoHeadroom })}
              label="Longer buffer when busy" />
          </SettingRow>
          <div className="grid gap-4 px-4 py-3 sm:grid-cols-2">
            <MiniSlider label="Index rate" value={voice.indexRate} min={0} max={1} step={0.05}
              onChange={(value) => onVoiceChange({ indexRate: value })} />
            <MiniSlider label="Protect" value={voice.protect} min={0} max={0.5} step={0.01}
              onChange={(value) => onVoiceChange({ protect: value })} />
            <MiniSlider label="Your loudness" value={voice.rmsMixRate} min={0} max={1} step={0.05}
              onChange={(value) => onVoiceChange({ rmsMixRate: value })} />
            <MiniSlider label="Pitch smoothing" value={voice.filterRadius} min={0} max={7} step={1}
              format={(v) => (v >= 3 ? String(Math.round(v)) : 'Off')}
              onChange={(value) => onVoiceChange({ filterRadius: value })} />
          </div>
          <dl className="divide-y divide-line">
            <Stat label="Round trip" value={latency > 0 ? ms(latency) : '—'} />
            <Stat label="Model" value={serverStats?.modelMs > 0 ? ms(serverStats.modelMs) : '—'} />
            <Stat label="Network" value={latency > 0 ? ms(Math.max(0, latency - (serverMs || 0))) : '—'} />
            <Stat label="GPU duty" value={serverStats?.activeModel ? `${Math.round(serverStats.inferenceDutyPercent || 0)}%` : '—'} />
          </dl>
          <div className="px-4 py-3">
            <StreamDiagnostics streamInfo={streamInfo} playbackStats={playbackStats} transport={transport}
              serverMs={serverMs} serverStats={serverStats} />
          </div>
        </More>
      </Column>

      <Column id="face" title="Face" Icon={ScanFace} accent="text-video-fg">
        <SettingsSection title="Swap">
          <SettingRow label="Face model" stacked>
            <Segmented label="Face model" hideLabel accent="face" size="sm" value={faceSettings.swapper || 'auto'}
              onChange={(value) => changeFace({ swapper: value })}
              options={[{ value: 'auto', label: 'Auto' }, ...(face.status?.swappers || []).map((item) => ({
                value: item.id, label: SWAPPER_CHOICES[item.id] || item.label,
                hint: `${SWAPPER_NAMES[item.id] || item.label}${swapMs[item.id] ? ` · ${Math.round(swapMs[item.id])} ms` : ''}`,
              }))]} />
          </SettingRow>
          <SettingRow label="Hands and hair in front">
            <Toggle checked={faceSettings.occlusion !== false} onChange={(occlusion) => changeFace({ occlusion })}
              label="Hands and hair in front" />
          </SettingRow>
        </SettingsSection>
        <SettingsSection title="Look">
          <SettingRow label="Your real mouth" stacked>
            <Segmented label="Your real mouth" hideLabel accent="face" size="sm" value={faceSettings.mouth || 'auto'}
              onChange={(value) => changeFace({ mouth: value })} options={AUTO_ON_OFF} />
          </SettingRow>
          <SettingRow label="Detail enhancer" stacked>
            <Segmented label="Detail enhancer" hideLabel accent="face" size="sm" value={faceSettings.enhance || 'auto'}
              onChange={(value) => changeFace({ enhance: value })} options={AUTO_ON_OFF} />
          </SettingRow>
        </SettingsSection>
        <More title="Performance">
          <dl className="divide-y divide-line">
            <Stat label="Shown" value={video.running ? `${video.shownFps} fps` : '—'} />
            <Stat label="Face swaps" value={stats ? `${stats.keys_per_second}/s` : '—'} />
            <Stat label="Face model" value={SWAPPER_NAMES[stats?.swapper] || stats?.swapper || '—'} />
            <Stat label="One swap" value={ms(stats?.swap_ms)} />
            <Stat label="GPU left by the voice" value={stats ? `${Math.round(stats.gpu_share * 100)}%` : '—'} />
            <Stat label="Tracking per frame" value={ms(stats?.track_ms)} />
            <Stat label="Blending per frame" value={ms(stats?.compose_ms)} />
            <Stat label="Voice delay" value={delayMs ? ms(delayMs) : '—'} />
            <Stat label="Shown after capture" value={video.running ? ms(video.shownDelay) : '—'} />
            <Stat label="Virtual camera" value={stats?.camera || (video.virtualCamera ? 'starting…' : 'off')} />
          </dl>
        </More>
      </Column>

      <Column id="general" title="General" Icon={Settings2} accent="text-brand-fg">
        <SettingsSection title="Appearance">
          <SettingRow label="Theme">
            <Segmented label="Theme" hideLabel options={THEMES} value={theme} onChange={onThemeChange} className="w-44" />
          </SettingRow>
        </SettingsSection>

        <section id="calls" className="rounded-lg border border-line bg-raised" aria-labelledby="calls-title">
          <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2">
            <h3 id="calls-title" className="text-xs font-semibold uppercase tracking-[0.12em] text-fg-subtle">Calls</h3>
            {missing ? (
              <Button variant="primary" onClick={calls.setUp} loading={calls.busy}>Set up</Button>
            ) : helpUrl && !calls.calls?.camera?.installed ? (
              <Button href={helpUrl} target="_blank" rel="noreferrer">Get OBS</Button>
            ) : null}
          </header>
          <div className="divide-y divide-line">
            <CallDevice Icon={Mic} label="Virtual microphone" item={calls.calls?.microphone} />
            <CallDevice Icon={Camera} label="Virtual camera" item={calls.calls?.camera} />
          </div>
          {callMessage ? (
            <p role="status" className={`border-t border-line px-4 py-2 text-xs ${callError ? 'text-danger-fg' : 'text-fg-subtle'}`}>
              {callMessage}
            </p>
          ) : null}
        </section>

        <SettingsSection title="Tests">
          <SettingRow label="Save my microphone with each take">
            <Toggle checked={captureSamples} onChange={onCaptureSamplesChange} label="Save my microphone with each take" />
          </SettingRow>
          <SettingRow label="Log">
            <Button icon={ScrollText} onClick={() => setLogsOpen(true)}>Open</Button>
          </SettingRow>
        </SettingsSection>

        <More title="Engine">
          <div className="px-4 py-3">
            <RuntimeReadiness config={config} checking={runtime.checking} error={runtime.error} onRefresh={runtime.refresh} />
          </div>
        </More>
        <p className="mt-auto px-1 text-xs text-fg-faint">{APP_NAME}{config.version ? ` ${config.version}` : ''}</p>
      </Column>

      {logsOpen ? (
        <Dialog title="Log" onClose={() => setLogsOpen(false)} width="max-w-5xl">
          <div className="flex h-[70svh] flex-col">
            <Suspense fallback={null}><LogsPanel active /></Suspense>
          </div>
        </Dialog>
      ) : null}
    </div>
  );
}

export default memo(Settings);
