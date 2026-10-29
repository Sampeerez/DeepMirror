import React, { memo, useMemo } from 'react';
import { Activity, RotateCcw } from 'lucide-react';
import Button from './Button';
import Segmented from './Segmented';
import SettingRow, { SettingsSection } from './SettingRow';
import StreamDiagnostics from './StreamDiagnostics';
import Toggle from './Toggle';
import { MiniSlider } from './Sliders';
import { groupedMethods, findF0Method } from '../lib/f0Methods';
import {
  PERFORMANCE_PROFILES, bufferDurationMs, profileForChunkSize, recommendPerformanceProfile,
} from '../lib/performance';
import { isRecommendedVoice } from '../lib/voice';

const SAMPLE_RATES = [44100, 48000];

// Rough end-to-end latency of a profile: about three chunks (capture, playback
// priming, server render) plus the server's context delays.
function latencyMs(profile, sampleRate) {
  return Math.round(3 * bufferDurationMs(profile.chunkSize, sampleRate) + profile.lookaheadSeconds * 1000 + 56);
}

/**
 * Voice › Settings: everything about the live voice that is set once, in one
 * place: latency, the pitch tracker, the silence saver and the model's detail,
 * with the stream's health beside it.
 */
function VoiceSettings({ config, onConfigChange, voice, onVoiceChange, onRestoreRecommended, running,
  latencyHistory = [], serverMs = 0, serverStats = {}, streamInfo = null, playbackStats = null, transport = {},
  f0Methods = [], streamSampleRate = null }) {
  const sampleRate = streamSampleRate || config.sampleRate;
  const profile = profileForChunkSize(config.chunkSize);
  const recommendation = useMemo(() => recommendPerformanceProfile({
    latencyHistory, sampleRate, serverMs, serverStats, isRunning: running,
  }), [latencyHistory, sampleRate, serverMs, serverStats, running]);
  const tracker = findF0Method(f0Methods, voice.f0Method);
  const liveGroups = groupedMethods(f0Methods, 'realtime');
  const recommended = isRecommendedVoice(voice);

  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-5 lg:col-span-2" aria-labelledby="voice-settings-title">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="panel-kicker text-ok-fg">Voice</p>
            <h2 id="voice-settings-title" className="panel-title">Voice settings</h2>
          </div>
          <Button icon={RotateCcw} onClick={onRestoreRecommended} disabled={recommended}
            title="This voice's own changes and effects are dropped">
            {recommended ? 'Automatic settings' : 'Back to automatic'}
          </Button>
        </div>
        <div className="mt-4 grid min-h-0 flex-1 auto-rows-min gap-4 overflow-y-auto pr-1 xl:grid-cols-2">
          <SettingsSection title="Delay and stability" className="xl:col-span-2"
            description={running ? 'Changes apply the next time you press Start.' : null}>
            <SettingRow label="Buffer" stacked value={`${Math.round(bufferDurationMs(config.chunkSize, sampleRate))} ms`}
              hint="Longer: steadier on a busy computer, a little more delay.">
              <Segmented label="Buffer" hideLabel accent="voice" size="sm" value={profile.id}
                onChange={(id) => onConfigChange({ chunkSize: PERFORMANCE_PROFILES.find((p) => p.id === id).chunkSize })}
                options={PERFORMANCE_PROFILES.filter((p) => p.id !== 'light').map((p) => ({
                  value: p.id, label: p.label, hint: `About ${latencyMs(p, sampleRate)} ms end to end`,
                }))} />
            </SettingRow>
            <SettingRow label="Longer buffer when busy" hint="Adds up to 85 ms instead of cutting the voice.">
              <Toggle checked={config.autoHeadroom} onChange={(autoHeadroom) => onConfigChange({ autoHeadroom })}
                label="Longer buffer when busy" />
            </SettingRow>
            <SettingRow label="Sample rate" hint="Match your microphone.">
              <Segmented label="Sample rate" hideLabel accent="voice" size="sm" value={config.sampleRate}
                onChange={(rate) => onConfigChange({ sampleRate: rate })}
                options={SAMPLE_RATES.map((rate) => ({ value: rate, label: `${rate / 1000} kHz` }))} className="w-48" />
            </SettingRow>
            <SettingRow label="Pitch tracker" htmlFor="voice-f0-method"
              hint={tracker.available ? tracker.description : tracker.reason}>
              <select id="voice-f0-method" value={tracker.id} onChange={(event) => onVoiceChange({ f0Method: event.target.value })}
                className="native-select-safe field w-48">
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
            {recommendation.ready ? (
              <SettingRow label="Measured on this computer"
                hint={`${recommendation.profile.label} fits (p95 ${Math.round(recommendation.p95Ms)} ms).`}>
                <Button icon={Activity} disabled={recommendation.profile.chunkSize === config.chunkSize}
                  onClick={() => onConfigChange({ chunkSize: recommendation.profile.chunkSize })}>
                  {recommendation.profile.chunkSize === config.chunkSize ? 'In use' : 'Use it'}
                </Button>
              </SettingRow>
            ) : null}
          </SettingsSection>

          <SettingsSection title="Silence saver" description="The model sleeps while you are quiet: less GPU, less heat.">
            <SettingRow label="Sleep when quiet">
              <Toggle checked={config.silenceSaver} onChange={(silenceSaver) => onConfigChange({ silenceSaver })}
                label="Sleep when quiet" />
            </SettingRow>
            <SettingRow label="Quiet below" stacked value={`${config.silenceThresholdDb} dB`}>
              <input type="range" aria-label="Quiet below" min="-80" max="-20" step="1" value={config.silenceThresholdDb}
                disabled={!config.silenceSaver} aria-valuetext={`${config.silenceThresholdDb} decibels`}
                onChange={(event) => onConfigChange({ silenceThresholdDb: Number(event.target.value) })}
                className="fx-slider w-full disabled:cursor-not-allowed disabled:opacity-45" />
            </SettingRow>
          </SettingsSection>

          <SettingsSection title="Model detail"
            description="Timbre (index), clean consonants (protect), whose loudness (0 yours, 1 the model's).">
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
          </SettingsSection>
        </div>
      </section>

      <section className="panel flex min-h-0 flex-col p-5" aria-labelledby="voice-health-title">
        <p className="panel-kicker text-ok-fg">Voice</p>
        <h2 id="voice-health-title" className="panel-title">Stream health</h2>
        <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
          <StreamDiagnostics streamInfo={streamInfo} playbackStats={playbackStats} transport={transport}
            serverMs={serverMs} serverStats={serverStats} />
        </div>
      </section>
    </div>
  );
}

export default memo(VoiceSettings);
