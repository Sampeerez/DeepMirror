import React, { memo } from 'react';
import Segmented from './Segmented';
import SettingRow, { SettingsSection } from './SettingRow';
import Toggle from './Toggle';

const SWAPPER_HINTS = {
  hyperswap_1b_256: 'Sharpest; hands and hair stay in front.',
  inswapper_128: 'Closest likeness, softer detail.',
  ghost_1_256: 'Fastest.',
};
const SHORT = { hyperswap_1b_256: 'HyperSwap', inswapper_128: 'inswapper', ghost_1_256: 'GHOST' };
const AUTO_ON_OFF = [
  { value: 'auto', label: 'Auto' },
  { value: 'on', label: 'Always' },
  { value: 'off', label: 'Never' },
];

function Stat({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-4 py-2 text-xs">
      <dt className="text-fg-subtle">{label}</dt>
      <dd className="font-mono tabular-nums text-fg-secondary">{value}</dd>
    </div>
  );
}

/** Face › Settings: how the face is cloned, and how fast it runs here. */
function VideoSettings({ face, video, delayMs }) {
  const settings = face.status?.settings || {};
  const swapMs = face.status?.swap_ms || {};
  const stats = video.stats;
  const change = (partial) => face.updateSettings(partial);
  const ms = (value) => (value == null ? '—' : `${Math.round(value)} ms`);
  const swapper = settings.swapper || 'auto';

  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-5 lg:col-span-2" aria-labelledby="video-settings-title">
        <p className="panel-kicker text-video-fg">Face</p>
        <h2 id="video-settings-title" className="panel-title">Face settings</h2>
        <div className="mt-4 grid min-h-0 flex-1 auto-rows-min gap-4 overflow-y-auto pr-1 xl:grid-cols-2">
          <SettingsSection title="The swap">
            <SettingRow label="Face model" stacked
              hint={swapper === 'auto' ? 'Picks the best one that fits next to the live voice on this computer.'
                : SWAPPER_HINTS[swapper]}>
              <Segmented label="Face model" hideLabel accent="face" size="sm" value={swapper}
                onChange={(value) => change({ swapper: value })}
                options={[{ value: 'auto', label: 'Auto' }, ...(face.status?.swappers || []).map((item) => ({
                  value: item.id, label: SHORT[item.id] || item.label,
                  hint: `${SWAPPER_HINTS[item.id] || ''}${swapMs[item.id] ? ` ${Math.round(swapMs[item.id])} ms here.` : ''}`,
                }))]} />
            </SettingRow>
            <SettingRow label="Hands and hair in front" hint="What covers your face stays in front of the new one.">
              <Toggle checked={settings.occlusion !== false} onChange={(occlusion) => change({ occlusion })}
                label="Hands and hair in front" />
            </SettingRow>
          </SettingsSection>

          <SettingsSection title="Look">
            <SettingRow label="Your real mouth" stacked
              hint="Your own lips over the new face, so every syllable moves at camera speed. Auto: when the GPU is busy.">
              <Segmented label="Your real mouth" hideLabel accent="face" size="sm" value={settings.mouth || 'auto'}
                onChange={(value) => change({ mouth: value })} options={AUTO_ON_OFF} />
            </SettingRow>
            <SettingRow label="Detail enhancer" stacked
              hint="Restores eyes, teeth and skin on close faces. Auto: when the GPU has room.">
              <Segmented label="Detail enhancer" hideLabel accent="face" size="sm" value={settings.enhance || 'auto'}
                onChange={(value) => change({ enhance: value })} options={AUTO_ON_OFF} />
            </SettingRow>
          </SettingsSection>

          <p className="rounded-lg border border-line bg-raised p-3 text-xs leading-5 text-fg-subtle xl:col-span-2">
            The voice always comes first: the face is swapped in the gaps the voice leaves on the GPU, and the
            frames in between follow your head exactly at camera speed. The video is held back by the voice&apos;s
            own delay, so your lips and the converted voice stay together, in the preview and on the virtual camera.
          </p>
        </div>
      </section>

      <section className="panel flex min-h-0 flex-col p-5" aria-labelledby="video-perf-title">
        <p className="panel-kicker text-video-fg">Face</p>
        <h2 id="video-perf-title" className="panel-title">Performance</h2>
        <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
          <SettingsSection title={video.running ? 'Live now' : 'Start the camera to measure'}>
            <dl className="divide-y divide-line">
              <Stat label="Shown" value={video.running ? `${video.shownFps} fps` : '—'} />
              <Stat label="Face swaps" value={stats ? `${stats.keys_per_second}/s` : '—'} />
              <Stat label="Face model" value={SHORT[stats?.swapper] || stats?.swapper || '—'} />
              <Stat label="One swap" value={ms(stats?.swap_ms)} />
              <Stat label="GPU left by the voice" value={stats ? `${Math.round(stats.gpu_share * 100)}%` : '—'} />
              <Stat label="Tracking per frame" value={ms(stats?.track_ms)} />
              <Stat label="Blending per frame" value={ms(stats?.compose_ms)} />
              <Stat label="Voice delay" value={delayMs ? ms(delayMs) : '—'} />
              <Stat label="Shown after capture" value={video.running ? ms(video.shownDelay) : '—'} />
              <Stat label="Virtual camera" value={stats?.camera || (video.virtualCamera ? 'starting…' : 'off')} />
            </dl>
          </SettingsSection>
          <p className="mt-3 text-[11px] leading-4 text-fg-faint">
            The virtual camera uses OBS Studio&apos;s driver: install OBS once (it does not need to be open), then pick
            &ldquo;OBS Virtual Camera&rdquo; in Discord, Meet or Teams.
          </p>
        </div>
      </section>
    </div>
  );
}

export default memo(VideoSettings);
