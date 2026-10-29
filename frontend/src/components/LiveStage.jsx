import React, { memo, useEffect, useRef } from 'react';
import { Pause, VideoOff } from 'lucide-react';
import { levelToPercent, nextPeak } from '../lib/meters';

// Solid zone coloring, VU convention: green / yellow / red. Listed as literal
// class names so Tailwind keeps them in the bundle.
const ZONE_CLASSES = ['bg-ok-solid', 'bg-warn-solid', 'bg-danger-solid'];

function zoneClass(percent) {
  if (percent > 82) return 'bg-danger-solid';
  if (percent > 60) return 'bg-warn-solid';
  return 'bg-ok-solid';
}

/**
 * A meter driven straight from the pipeline's meter store.
 *
 * Levels change ~60x per second. Routing them through React state would
 * re-render the whole App on every animation frame, so the bar, the peak-hold
 * marker and the readout are written to the DOM directly instead. The visual
 * behaviour, 3x gain, zone colours, slow peak decay, is unchanged.
 */
function VuMeter({ label, meters, channel }) {
  const barRef = useRef(null);
  const peakRef = useRef(null);
  const readoutRef = useRef(null);
  const peakValueRef = useRef(0);

  useEffect(() => {
    if (!meters?.subscribe) return undefined;

    return meters.subscribe((value) => {
      const level = value?.[channel] ?? 0;
      const percent = levelToPercent(level);
      peakValueRef.current = nextPeak(peakValueRef.current, percent);

      const bar = barRef.current;
      if (bar) {
        bar.style.width = `${percent}%`;
        const nextClass = zoneClass(percent);
        for (const cls of ZONE_CLASSES) {
          bar.classList.toggle(cls, cls === nextClass);
        }
      }

      const peak = peakRef.current;
      if (peak) {
        const visible = peakValueRef.current > 2;
        peak.style.opacity = visible ? '1' : '0';
        if (visible) peak.style.left = `calc(${peakValueRef.current}% - 1px)`;
      }

      const readout = readoutRef.current;
      if (readout) readout.textContent = `${(level * 100).toFixed(1)}%`;
    });
  }, [meters, channel]);

  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="w-7 shrink-0 text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-subtle">{label}</span>
      <div className="relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-meter-track">
        <div
          ref={barRef}
          className="h-full bg-ok-solid transition-all duration-75 ease-out"
          style={{ width: '0%' }}
        />
        <div
          ref={peakRef}
          className="absolute top-0 h-full w-[2px] bg-meter-peak"
          style={{ left: '-1px', opacity: 0 }}
        />
      </div>
      <span ref={readoutRef} className="w-11 shrink-0 text-right font-mono text-[10px] tabular-nums text-fg-faint">0.0%</span>
    </div>
  );
}

function StateBadge({ bypass, serverStats }) {
  if (bypass) {
    return (
      <span className="rounded-sm border border-warn-line-strong bg-warn-bg px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-warn-fg">
        Original
      </span>
    );
  }
  if (serverStats?.inferenceSleeping) {
    return (
      <span className="inline-flex items-center gap-1 rounded-sm border border-ok-line bg-ok-bg px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-ok-fg">
        <Pause className="h-3 w-3" aria-hidden="true" />
        Saver
      </span>
    );
  }
  return null;
}

/**
 * Live's centre: the camera and the sound together. The face bar on top, the
 * cloned video filling the middle, and under it, always, the spectrum, the
 * levels, the latency and the record button.
 */
function LiveStage({
  faceBar = null, videoPreview = null, videoNote = null, visualizer = null, meters, latency, serverStats,
  bypass = false, record = null,
}) {
  const latencyMs = Math.round(latency);
  const latencyColor = latencyMs <= 0 ? 'text-fg-subtle'
    : latencyMs < 60 ? 'text-ok-fg' : latencyMs < 150 ? 'text-warn-fg' : 'text-danger-fg';

  return (
    <section className="panel flex min-h-0 flex-col gap-3 p-4" aria-labelledby="live-face-title">
      {faceBar}
      <div className="relative flex min-h-48 flex-1 items-center justify-center overflow-hidden rounded-md bg-black">
        {videoPreview || (
          <div className="flex flex-col items-center gap-2 text-fg-faint">
            <VideoOff className="h-8 w-8" strokeWidth={1.5} aria-hidden="true" />
            <span className="text-xs">Camera off</span>
          </div>
        )}
        {videoNote ? (
          <p role="status" className={`absolute inset-x-3 bottom-2 truncate text-center text-xs ${
            videoNote.tone === 'danger' ? 'text-danger-fg' : videoNote.tone === 'warn' ? 'text-warn-fg' : 'text-fg-subtle'}`}>
            {videoNote.text}
          </p>
        ) : null}
      </div>

      {visualizer}
      {serverStats?.processingError && !bypass ? (
        <p role="alert" className="truncate text-xs text-danger-fg" title={serverStats.processingError}>
          {serverStats.processingError}
        </p>
      ) : null}
      <div className="grid gap-x-5 gap-y-1.5 sm:grid-cols-2">
        <VuMeter label="In" meters={meters} channel="input" />
        <VuMeter label="Out" meters={meters} channel="output" />
      </div>

      <div className="flex min-h-9 items-center gap-3 border-t border-line pt-3">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-subtle">Latency</span>
        <span className={`font-mono text-xs font-semibold tabular-nums ${latencyColor}`}>
          {latencyMs > 0 ? `${latencyMs} ms` : '--'}
        </span>
        <StateBadge bypass={bypass} serverStats={serverStats} />
        <div className="ml-auto min-w-0">{record}</div>
      </div>
    </section>
  );
}

export default memo(LiveStage);
