import React, { useId } from 'react';

/**
 * One choice among a few, all visible: replaces a drop-down of 2-5 options.
 *
 * A radiogroup (WAI-ARIA): arrow keys move and select, only the chosen option
 * is in the Tab order, and each option keeps its text so nothing depends on
 * an icon or a colour. ``accent`` tints the chosen option ("voice" or "face").
 */
export default function Segmented({ label, options, value, onChange, disabled = false, accent = null,
  size = 'md', className = '', hideLabel = false }) {
  const labelId = useId();
  const accentText = accent === 'face' ? 'text-video-fg' : accent === 'voice' ? 'text-ok-fg' : 'text-fg';

  const move = (event) => {
    const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    if (!(event.key in keys) && event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault();
    const enabled = options.filter((option) => !option.disabled);
    const index = Math.max(0, enabled.findIndex((option) => option.value === value));
    const next = event.key === 'Home' ? enabled[0]
      : event.key === 'End' ? enabled[enabled.length - 1]
        : enabled[(index + keys[event.key] + enabled.length) % enabled.length];
    if (!next) return;
    onChange(next.value);
    event.currentTarget.querySelector(`[data-value="${CSS.escape(String(next.value))}"]`)?.focus();
  };

  return (
    <div className={className}>
      {label ? (
        <p id={labelId} className={hideLabel ? 'sr-only' : 'mb-1.5 text-xs font-medium text-fg-muted'}>{label}</p>
      ) : null}
      <div
        role="radiogroup"
        aria-labelledby={label ? labelId : undefined}
        aria-disabled={disabled || undefined}
        onKeyDown={disabled ? undefined : move}
        className="flex w-full items-stretch gap-1 rounded-lg border border-[color:var(--frost-border)] bg-sunken p-1"
      >
        {options.map((option) => {
          const selected = option.value === value;
          const Icon = option.icon;
          return (
            <button
              key={String(option.value)}
              type="button"
              role="radio"
              aria-checked={selected}
              tabIndex={selected ? 0 : -1}
              disabled={disabled || option.disabled}
              data-value={String(option.value)}
              data-selected={selected}
              title={option.hint}
              onClick={() => onChange(option.value)}
              className={`flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45 ${
                size === 'sm' ? 'h-7' : 'h-8'
              } ${selected
                ? `frost-control ${accentText}`
                : 'border-transparent text-fg-muted hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]'}`}
            >
              {Icon ? <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" /> : null}
              <span className="min-w-0 truncate">{option.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
