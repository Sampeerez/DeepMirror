import React, { memo } from 'react';

const SWAPPER_NOTES = {
  hyperswap_1b_256: 'sharpest; keeps hands and hair in front',
  inswapper_128: 'closest likeness, softer detail',
  ghost_1_256: 'fastest',
};

function Field({ label, htmlFor, hint, children }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-fg-muted">{label}</label>
      <div className="mt-1.5">{children}</div>
      {hint ? <p className="mt-1 text-[11px] leading-4 text-fg-faint">{hint}</p> : null}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line py-2 text-xs last:border-b-0">
      <dt className="text-fg-subtle">{label}</dt>
      <dd className="font-mono tabular-nums text-fg-secondary">{value}</dd>
    </div>
  );
}

/** Video › Video settings: how the face is cloned, and how fast it runs here. */
function VideoSettings({ face, video, delayMs }) {
  const settings = face.status?.settings || {};
  const swapMs = face.status?.swap_ms || {};
  const stats = video.stats;
  const change = (partial) => face.updateSettings(partial);
  const ms = (value) => (value == null ? '—' : `${Math.round(value)} ms`);

  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-5 lg:col-span-2" aria-labelledby="video-settings-title">
        <p className="panel-kicker text-video-fg">Video</p>
        <h2 id="video-settings-title" className="panel-title">Video settings</h2>
        <div className="mt-4 grid min-h-0 flex-1 auto-rows-min gap-x-4 gap-y-4 overflow-y-auto pr-1 sm:grid-cols-2">
          <Field label="Face swapper" htmlFor="video-swapper"
            hint="Automatic picks the best one that fits next to the live voice on this computer.">
            <select id="video-swapper" className="native-select-safe field" value={settings.swapper || 'auto'}
              onChange={(event) => change({ swapper: event.target.value })}>
              <option value="auto">Automatic (recommended)</option>
              {(face.status?.swappers || []).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}{SWAPPER_NOTES[item.id] ? `: ${SWAPPER_NOTES[item.id]}` : ''}
                  {swapMs[item.id] ? ` (${Math.round(swapMs[item.id])} ms)` : ''}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Your real mouth" htmlFor="video-mouth"
            hint="Your own lips over the new face: every syllable moves at camera speed. Automatic when the GPU is busy.">
            <select id="video-mouth" className="native-select-safe field" value={settings.mouth || 'auto'}
              onChange={(event) => change({ mouth: event.target.value })}>
              <option value="auto">Automatic (recommended)</option>
              <option value="on">Always</option>
              <option value="off">Never</option>
            </select>
          </Field>
          <Field label="Detail enhancer" htmlFor="video-enhance"
            hint="Restores eyes, teeth and skin on close faces when the GPU has room.">
            <select id="video-enhance" className="native-select-safe field" value={settings.enhance || 'auto'}
              onChange={(event) => change({ enhance: event.target.value })}>
              <option value="auto">Automatic (recommended)</option>
              <option value="on">Always</option>
              <option value="off">Never</option>
            </select>
          </Field>
          <Field label="Hands and hair in front" htmlFor="video-occlusion"
            hint="What covers your face stays in front of the new one.">
            <select id="video-occlusion" className="native-select-safe field"
              value={settings.occlusion === false ? 'off' : 'on'}
              onChange={(event) => change({ occlusion: event.target.value === 'on' })}>
              <option value="on">Keep them in front (recommended)</option>
              <option value="off">Ignore them (faster)</option>
            </select>
          </Field>
          <div className="sm:col-span-2 rounded-md border border-line bg-raised p-3 text-xs leading-5 text-fg-subtle">
            The voice always comes first: the face is swapped in the gaps the voice leaves on the GPU, and the
            frames in between follow your head exactly at camera speed. The video is held back by the voice&apos;s
            own delay, so your lips and the converted voice stay together, in the preview and on the virtual camera.
          </div>
        </div>
      </section>

      <section className="panel flex min-h-0 flex-col p-5" aria-labelledby="video-perf-title">
        <p className="panel-kicker text-video-fg">Video</p>
        <h2 id="video-perf-title" className="panel-title">Performance</h2>
        <dl className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
          <Stat label="Shown" value={video.running ? `${video.shownFps} fps` : '—'} />
          <Stat label="Face swaps" value={stats ? `${stats.keys_per_second}/s` : '—'} />
          <Stat label="Swapper" value={stats?.swapper || '—'} />
          <Stat label="One swap" value={ms(stats?.swap_ms)} />
          <Stat label="GPU left by the voice" value={stats ? `${Math.round(stats.gpu_share * 100)}%` : '—'} />
          <Stat label="Tracking per frame" value={ms(stats?.track_ms)} />
          <Stat label="Blending per frame" value={ms(stats?.compose_ms)} />
          <Stat label="Voice delay" value={delayMs ? ms(delayMs) : '—'} />
          <Stat label="Video shown after capture" value={video.running ? ms(video.shownDelay) : '—'} />
          <Stat label="Virtual camera" value={stats?.camera || (video.virtualCamera ? 'starting…' : 'off')} />
        </dl>
        <p className="mt-3 text-[11px] leading-4 text-fg-faint">
          The virtual camera uses OBS Studio&apos;s driver: install OBS once (it does not need to be open), then pick
          &ldquo;OBS Virtual Camera&rdquo; in Discord, Meet or Teams.
        </p>
      </section>
    </div>
  );
}

export default memo(VideoSettings);
