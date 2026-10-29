import React, { memo, useCallback } from 'react';
import Button from './Button';
import Toggle from './Toggle';
import { EFFECT_DEFS, countActiveEffects, defaultEffects } from '../lib/effects';

function formatValue(param, value) {
  const num = Number(value);
  const text = param.step >= 1 ? String(Math.round(num)) : num.toFixed(2).replace(/\.?0+$/, '');
  return `${text}${param.unit ? ` ${param.unit}` : ''}`;
}

// Memoized so dragging one card's slider repaints that card only, not the
// other eleven. Parameters show only while the effect is on, so the list of
// effects stays short and scannable.
const EffectCard = memo(function EffectCard({ def, state, onChange }) {
  const enabled = Boolean(state?.enabled);

  return (
    <div className={`rounded-md border px-3 py-2.5 transition-colors duration-150 ${
      enabled ? 'border-ok-line bg-ok-bg' : 'border-line bg-raised hover:border-line-hover'}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className={`truncate text-xs font-semibold ${enabled ? 'text-fg' : 'text-fg-muted'}`}>{def.label}</p>
        </div>
        <Toggle checked={enabled} label={`Toggle ${def.label}`}
          onChange={(next) => onChange(def.key, { ...state, enabled: next })} />
      </div>

      {enabled && def.params.length > 0 ? (
        <div className="mt-2.5 space-y-2">
          {def.params.map((param) => (
            <div key={param.key}>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-[11px] font-medium text-fg-subtle">{param.label}</span>
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-fg-muted">
                  {formatValue(param, state?.[param.key] ?? param.default)}
                </span>
              </div>
              <input type="range" aria-label={`${def.label} ${param.label}`} min={param.min} max={param.max}
                step={param.step} value={state?.[param.key] ?? param.default}
                onChange={(event) => onChange(def.key, { ...state, [param.key]: Number(event.target.value) })}
                className="fx-slider mt-1 w-full" />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
});

// `onEffectsChange` accepts either the next rack or an updater over the
// previous one (it is forwarded to a React state setter). The updater form
// keeps this callback free of `effects`, and therefore stable for the cards.
function EffectsRack({ effects, formantShift = 0, onEffectsChange }) {
  const activeCount = countActiveEffects(effects, formantShift);

  const handleChange = useCallback((key, entry) => {
    onEffectsChange((previous) => ({ ...previous, [key]: entry }));
  }, [onEffectsChange]);

  return (
    <div id="effects" className="space-y-3">
      <div className="flex h-8 items-center justify-between gap-3">
        <p className="min-w-0 truncate text-[11px] text-fg-subtle">
          {activeCount > 0 ? `${activeCount} on` : 'None on'}
        </p>
        <Button onClick={() => onEffectsChange(defaultEffects())} disabled={activeCount === 0} className="w-24">
          All off
        </Button>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {EFFECT_DEFS.map((def) => (
          <EffectCard key={def.key} def={def} state={effects[def.key]} onChange={handleChange} />
        ))}
      </div>
    </div>
  );
}

export default memo(EffectsRack);
