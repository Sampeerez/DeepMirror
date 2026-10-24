import React from 'react';
import { RotateCcw } from 'lucide-react';
import Button from './Button';
import Toggle from './Toggle';

/** A main control (pitch, formant): big readout, reset button, scale under the track. */
export function BigSlider({ label, value, min, max, step, unit, onChange, onReset, resetDisabled, accessory, hint }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <p className="truncate text-xs font-medium text-fg-muted">{label}</p>
          {accessory}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className={`w-20 text-right font-mono text-lg font-semibold tabular-nums ${value === 0 ? 'text-fg-subtle' : 'text-fg'}`}>
            {value > 0 ? '+' : ''}{Number(value).toFixed(1)}{unit}
          </span>
          <Button icon={RotateCcw} variant="ghost" onClick={onReset} disabled={resetDisabled ?? value === 0}
            aria-label={`Reset ${label}`} title={`Reset ${label}`} />
        </div>
      </div>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value}
        onChange={(event) => onChange(Number(event.target.value))} className="mt-2 w-full" />
      <div className="mt-1 flex items-center justify-between text-[10px] font-medium text-fg-faint">
        <span>{min}</span>
        <span>0</span>
        <span>+{max}</span>
      </div>
      {hint ? <p className="mt-1 text-[11px] leading-4 text-fg-subtle">{hint}</p> : null}
    </div>
  );
}

/** A secondary control: label and value on one line, slim track. */
export function MiniSlider({ label, value, min, max, step, onChange, format = (v) => v.toFixed(2), hint, disabled }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <span className="min-w-0 truncate text-xs font-medium text-fg-muted" title={label}>{label}</span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-fg-secondary">{format(Number(value))}</span>
      </div>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))} className="fx-slider mt-2 w-full" />
      {hint ? <p className="mt-1 text-[11px] leading-4 text-fg-subtle">{hint}</p> : null}
    </div>
  );
}

/** A setting that is on or off, with one line explaining it. */
export function SwitchRow({ label, hint, checked, onChange, children }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <p className="text-xs font-medium text-fg-muted">{label}</p>
        {hint ? <p className="mt-0.5 text-[11px] leading-4 text-fg-subtle">{hint}</p> : null}
      </div>
      {children}
      <Toggle checked={checked} onChange={onChange} label={label} />
    </div>
  );
}
