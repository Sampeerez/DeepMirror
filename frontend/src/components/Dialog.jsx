import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import Button from './Button';

/**
 * A modal over the studio: Escape or the backdrop closes it, focus stays
 * inside while it is open and returns to the opener afterwards, and the body
 * scrolls inside the dialog, never the page behind it.
 */
export default function Dialog({ title, onClose, children, width = 'max-w-lg' }) {
  const ref = useRef(null);

  useEffect(() => {
    const opener = document.activeElement;
    ref.current?.querySelector('button, select, input')?.focus({ preventScroll: true });
    const onKeyDown = (event) => { if (event.key === 'Escape') onClose(); };
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = overflow;
      window.removeEventListener('keydown', onKeyDown);
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, [onClose]);

  const trapFocus = (event) => {
    if (event.key !== 'Tab') return;
    const controls = [...ref.current.querySelectorAll('button, input, select, a[href], [tabindex]')]
      .filter((element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length > 0);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6">
      <button type="button" className="absolute inset-0 bg-scrim backdrop-blur-md" onClick={onClose}
        aria-label={`Close ${title}`} tabIndex={-1} />
      <section ref={ref} role="dialog" aria-modal="true" aria-label={title} onKeyDown={trapFocus}
        className={`relative flex max-h-[calc(100svh-3rem)] w-full ${width} flex-col rounded-lg border border-line-strong bg-overlay shadow-[0_16px_64px_var(--overlay-shadow)]`}>
        <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-3">
          <h2 className="min-w-0 truncate text-base font-semibold text-fg">{title}</h2>
          <Button icon={X} variant="ghost" size="md" onClick={onClose} aria-label="Close" title="Close" />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </section>
    </div>
  );
}
