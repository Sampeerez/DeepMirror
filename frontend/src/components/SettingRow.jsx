import React from 'react';

/** A group of settings rows in one box. */
export function SettingsSection({ children, className = '' }) {
  return <section className={`divide-y divide-line rounded-lg border border-line bg-raised ${className}`}>{children}</section>;
}

/**
 * One setting, always the same shape so every control lines up: its icon and
 * name (and current value, right) on top, the control under it at full width,
 * 36 px tall whatever it is (a choice, a slider, a picker).
 */
export default function SettingRow({ label, icon: Icon = null, htmlFor, children, value = null, hint = null }) {
  const LabelTag = htmlFor ? 'label' : 'p';
  return (
    <div className="px-3 py-2.5">
      <div className="mb-2 flex h-4 items-center gap-2">
        {Icon ? <Icon className="h-3.5 w-3.5 shrink-0 text-fg-subtle" aria-hidden="true" /> : null}
        <LabelTag htmlFor={htmlFor} className="min-w-0 flex-1 truncate text-xs font-medium text-fg-secondary">{label}</LabelTag>
        {value != null ? <span className="shrink-0 font-mono text-[11px] tabular-nums text-fg-muted">{value}</span> : null}
      </div>
      <div className="flex min-h-9 items-center">{children}</div>
      {hint ? <p className="mt-1.5 text-xs text-warn-fg">{hint}</p> : null}
    </div>
  );
}
