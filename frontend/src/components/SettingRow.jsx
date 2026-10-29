import React from 'react';

/** A titled group of settings rows. */
export function SettingsSection({ title, description = null, children, className = '' }) {
  return (
    <section className={`rounded-lg border border-line bg-raised ${className}`}>
      <header className="border-b border-line px-4 py-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-fg-subtle">{title}</h3>
        {description ? <p className="mt-0.5 text-[11px] text-fg-faint">{description}</p> : null}
      </header>
      <div className="divide-y divide-line">{children}</div>
    </section>
  );
}

/**
 * One setting: what it is on the left (with a one-line explanation), its
 * control on the right. ``stacked`` puts a wide control (a slider, a choice of
 * four) under the text instead.
 */
export default function SettingRow({ label, hint = null, htmlFor, children, stacked = false, value = null }) {
  const LabelTag = htmlFor ? 'label' : 'p';
  return (
    <div className={`px-4 py-2.5 ${stacked ? '' : 'flex items-center justify-between gap-4'}`}>
      <div className="min-w-0">
        <div className="flex items-baseline justify-between gap-3">
          <LabelTag htmlFor={htmlFor} className="text-sm font-medium text-fg-secondary">{label}</LabelTag>
          {value != null ? <span className="font-mono text-xs tabular-nums text-fg-muted">{value}</span> : null}
        </div>
        {hint ? <p className="mt-0.5 text-[11px] leading-4 text-fg-faint">{hint}</p> : null}
      </div>
      <div className={stacked ? 'mt-2' : 'shrink-0'}>{children}</div>
    </div>
  );
}
