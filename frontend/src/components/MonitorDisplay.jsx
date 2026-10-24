import React, { memo, useEffect, useMemo, useRef } from 'react';
import { Pause } from 'lucide-react';
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
 * behaviour — 3x gain, zone colours, slow peak decay — is unchanged.
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
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-fg-subtle">{label}</span>
        <span
          ref={readoutRef}
          className="font-mono text-[10px] uppercase tracking-[0.1em] text-fg-faint"
        >
          0.0%
        </span>
      </div>
      <div className="relative h-2 bg-meter-track">
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
    </div>
  );
}

function Sparkline({ history }) {
  const path = useMemo(() => {
    if (!history || history.length < 2) return '';
    const w = 100;
    const h = 28;
    const max = Math.max(...history, 50);
    const points = history.map((value, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - Math.min(value / max, 1) * (h - 2) - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return `M${points.join(' L')}`;
  }, [history]);

  if (!path) {
    return <div className="h-[28px]" />;
  }

  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="h-[28px] w-full text-spark">
      <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function StatChip({ label, value }) {
  return (
    <div className="min-w-0 rounded-sm border border-line bg-input px-2 py-1.5 text-center">
      <p className="truncate text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-subtle">{label}</p>
      <p className="mt-0.5 truncate font-mono text-xs font-semibold tabular-nums text-fg-secondary">{value}</p>
    </div>
  );
}

/** The Monitor card: spectrum, levels, latency and the recorder. */
function MonitorDisplay({
  meters,
  latency,
  latencyHistory,
  serverMs,
  serverStats,
  bypass = false,
  visualizer = null,
  recorder = null,
}) {
  const latencyMs = Math.round(latency);
  const networkMs = Math.max(0, Math.round(latency - (serverMs || 0)));
  const latencyColor =
    latencyMs <= 0
      ? 'text-fg-subtle'
      : latencyMs < 60
        ? 'text-ok-fg'
        : latencyMs < 150
          ? 'text-warn-fg'
          : 'text-danger-fg';

  return (
    <section className="panel flex h-full min-h-0 flex-col p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="panel-kicker">Monitor</p>
          <h2 className="panel-title truncate">Levels & recorder</h2>
        </div>
        {bypass ? (
          <span className="rounded-sm border border-warn-line-strong bg-warn-bg px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-warn-fg">
            Bypassed
          </span>
        ) : serverStats?.inferenceSleeping ? (
          <span className="inline-flex items-center gap-1.5 rounded-sm border border-ok-line bg-ok-bg px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-ok-fg">
            <Pause className="h-3 w-3" aria-hidden="true" />
            Saver
          </span>
        ) : serverStats?.effectsActive > 0 ? (
          <span className="rounded-sm border border-line-strong bg-control px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-secondary">
            {serverStats.effectsActive} FX
          </span>
        ) : null}
      </div>

      <div className="mt-4 shrink-0 space-y-3">
        {visualizer}
        {serverStats?.processingError && !bypass ? (
          <p role="alert" className="truncate text-xs text-danger-fg" title={serverStats.processingError}>
            {serverStats.processingError}
          </p>
        ) : null}
        <VuMeter label="Input" meters={meters} channel="input" />
        <VuMeter label="Output" meters={meters} channel="output" />

        <div className="grid grid-cols-[minmax(0,1.3fr)_repeat(3,minmax(0,1fr))] items-stretch gap-2">
          <div className="min-w-0 rounded-sm border border-line bg-input px-2 py-1.5">
            <div className="flex items-baseline justify-between gap-2">
              <p className="truncate text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-subtle">Latency</p>
              <p className={`shrink-0 font-mono text-xs font-semibold tabular-nums ${latencyColor}`}>
                {latencyMs > 0 ? `${latencyMs} ms` : '--'}
              </p>
            </div>
            <Sparkline history={latencyHistory} />
          </div>
          <StatChip label="Model" value={serverStats?.modelMs > 0 ? `${Math.round(serverStats.modelMs)} ms` : '--'} />
          <StatChip label="Network" value={latencyMs > 0 ? `${networkMs} ms` : '--'} />
          <StatChip label="Duty"
            value={serverStats?.activeModel ? `${Math.round(serverStats.inferenceDutyPercent || 0)}%` : '--'} />
        </div>
      </div>

      {recorder ? <div className="mt-4 flex min-h-0 flex-1 flex-col border-t border-line pt-3">{recorder}</div> : null}
    </section>
  );
}

export default memo(MonitorDisplay);
