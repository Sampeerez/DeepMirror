import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { readStored, writeStored } from '../lib/storage';

const STORAGE_KEY = 'ovc_sections_v1';

function readOpen(id, fallback) {
  const stored = readStored(STORAGE_KEY);
  return typeof stored?.[id] === 'boolean' ? stored[id] : fallback;
}

/**
 * A section that folds to one line: its title and a short summary of what is
 * inside ("RMVPE · Ready"). The open/closed state is remembered per section.
 */
export default function Collapsible({ id, title, summary, defaultOpen = false, actions = null, children, className = '' }) {
  const [open, setOpen] = useState(() => readOpen(id, defaultOpen));
  const toggle = () => {
    setOpen((value) => {
      writeStored(STORAGE_KEY, { ...(readStored(STORAGE_KEY) || {}), [id]: !value });
      return !value;
    });
  };
  const bodyId = `section-${id}`;
  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        <button type="button" onClick={toggle} aria-expanded={open} aria-controls={bodyId}
          className="frost-control flex min-h-9 min-w-0 flex-1 items-center justify-between gap-3 px-3 py-2 text-left">
          <span className="min-w-0">
            <span className="block text-xs font-medium text-fg-secondary">{title}</span>
            {summary && !open ? <span className="block truncate text-[11px] text-fg-subtle">{summary}</span> : null}
          </span>
          <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-fg-subtle transition-transform ${open ? 'rotate-180' : ''}`}
            aria-hidden="true" />
        </button>
        {actions}
      </div>
      {open ? <div id={bodyId} className="mt-3">{children}</div> : null}
    </div>
  );
}
