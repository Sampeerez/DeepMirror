import React, { useEffect, useId, useRef, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';

/**
 * A compact chip that shows the current choice and opens a short list:
 * microphones, speakers, cameras, voices, faces. Nothing is open until asked.
 *
 * Button + listbox (WAI-ARIA): Enter/Space/ArrowDown open it, arrows move,
 * Enter chooses, Escape or a click outside closes and returns focus.
 */
export default function Picker({ label, icon: Icon, value, options, onChange, disabled = false, accent = null,
  placeholder = 'Choose…', className = '', render = null }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef(null);
  const buttonRef = useRef(null);
  const listId = useId();
  const current = options.find((option) => option.value === value);
  const accentText = accent === 'face' ? 'text-video-fg' : accent === 'voice' ? 'text-ok-fg' : 'text-fg-subtle';

  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => { if (!rootRef.current?.contains(event.target)) setOpen(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, [open]);

  const openList = () => {
    setActive(Math.max(0, options.findIndex((option) => option.value === value)));
    setOpen(true);
  };
  const choose = (option) => {
    if (option.disabled) return;
    onChange(option.value);
    setOpen(false);
    buttonRef.current?.focus();
  };
  const onKeyDown = (event) => {
    if (!open) {
      if (['Enter', ' ', 'ArrowDown'].includes(event.key)) { event.preventDefault(); openList(); }
      return;
    }
    if (event.key === 'Escape') { event.preventDefault(); setOpen(false); buttonRef.current?.focus(); }
    else if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => Math.min(options.length - 1, i + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => Math.max(0, i - 1)); }
    else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (options[active]) choose(options[active]); }
    else if (event.key === 'Tab') setOpen(false);
  };

  return (
    <div ref={rootRef} className={`relative min-w-0 ${className}`} onKeyDown={onKeyDown}>
      <button
        ref={buttonRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-label={`${label}: ${current?.label || placeholder}`}
        title={current?.label || placeholder}
        onClick={() => (open ? setOpen(false) : openList())}
        className="frost-control flex h-9 w-full min-w-0 items-center gap-2 px-3 text-left text-xs disabled:cursor-not-allowed disabled:opacity-50"
      >
        {Icon ? <Icon className={`h-4 w-4 shrink-0 ${accentText}`} aria-hidden="true" /> : null}
        <span className="min-w-0 flex-1">
          <span className="block text-[10px] font-semibold uppercase leading-3 tracking-[0.12em] text-fg-faint">{label}</span>
          <span className="block truncate font-medium text-fg-secondary">{current?.label || placeholder}</span>
        </span>
        <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-fg-faint transition-transform ${open ? 'rotate-180' : ''}`}
          aria-hidden="true" />
      </button>
      {open ? (
        <ul id={listId} role="listbox" aria-label={label}
          aria-activedescendant={`${listId}-${active}`}
          className="absolute left-0 right-0 top-[calc(100%+4px)] z-30 max-h-72 overflow-y-auto rounded-lg border border-line-strong bg-overlay p-1 shadow-[0_12px_32px_var(--overlay-shadow)]">
          {options.map((option, index) => {
            const selected = option.value === value;
            return (
              <li
                key={String(option.value)}
                id={`${listId}-${index}`}
                role="option"
                aria-selected={selected}
                aria-disabled={option.disabled || undefined}
                onPointerEnter={() => setActive(index)}
                onClick={() => choose(option)}
                className={`flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-2 text-xs ${
                  index === active ? 'bg-control-hover text-fg' : 'text-fg-secondary'
                } ${option.disabled ? 'cursor-not-allowed opacity-45' : ''}`}
              >
                {render ? render(option) : <span className="min-w-0 flex-1 truncate">{option.label}</span>}
                {option.note ? <span className="shrink-0 text-[10px] text-fg-faint">{option.note}</span> : null}
                <Check className={`h-3.5 w-3.5 shrink-0 ${selected ? accentText : 'invisible'}`} aria-hidden="true" />
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
