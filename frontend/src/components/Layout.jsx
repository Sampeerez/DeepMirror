import React, { useRef } from 'react';
import { AudioWaveform, Boxes, FileDown, GraduationCap, Mic2, ScrollText, UserRound } from 'lucide-react';
import { APP_NAME, APP_TAGLINE } from '../lib/brand';

// Ids match lib/navigation TAB_IDS; the matching panels in App carry
// id="panel-<id>" and aria-labelledby="tab-<id>".
const TABS = [
  {
    id: 'studio',
    label: 'Studio',
    Icon: Mic2,
  },
  {
    id: 'models',
    label: 'Models',
    Icon: Boxes,
  },
  {
    id: 'voice',
    label: 'My voice',
    Icon: UserRound,
  },
  {
    id: 'training',
    label: 'Train',
    Icon: GraduationCap,
  },
  {
    id: 'converter',
    label: 'Converter',
    Icon: FileDown,
  },
  {
    id: 'logs',
    label: 'Logs',
    Icon: ScrollText,
  },
];

/**
 * Top-level view switcher.
 *
 * Marked up as a tablist (not a nav) because the three views are panels of one
 * document rather than separate pages: assistive tech announces "tab 2 of 3,
 * selected", and arrow keys move between them while only the selected tab
 * sits in the Tab order, as the WAI-ARIA pattern prescribes.
 */
function TabList({ tab, onTabChange }) {
  const listRef = useRef(null);

  const handleKeyDown = (event) => {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();

    const currentIndex = Math.max(0, TABS.findIndex(({ id }) => id === tab));
    const nextIndex = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? TABS.length - 1
        : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length;
    const nextId = TABS[nextIndex].id;

    onTabChange?.(nextId);
    listRef.current?.querySelector(`[data-tab="${nextId}"]`)?.focus();
  };

  return (
    <div
      ref={listRef}
      role="tablist"
      aria-label="Workspace"
      onKeyDown={handleKeyDown}
      className="order-3 flex w-full items-center gap-1 rounded-lg border border-[color:var(--frost-border)] bg-sunken p-1 sm:order-none sm:w-auto"
    >
      {TABS.map((item) => {
        const selected = tab === item.id;
        const Icon = item.Icon;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`tab-${item.id}`}
            aria-selected={selected}
            aria-controls={`panel-${item.id}`}
            tabIndex={selected ? 0 : -1}
            data-tab={item.id}
            data-selected={selected}
            onClick={() => onTabChange?.(item.id)}
            className={`flex h-8 min-w-0 flex-1 items-center justify-center gap-1.5 whitespace-nowrap rounded-md border px-2 text-xs font-medium sm:flex-initial sm:gap-2 sm:px-4 ${
              selected
                ? 'frost-control text-fg'
                : 'border-transparent text-fg-muted transition-colors duration-150 ease-out hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]'
            }`}
          >
            <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The page frame. On desktop it is exactly one screen tall: the header, then
 * the active tab, which fills the rest and scrolls inside its own panels if
 * it must, never the page. Narrow screens stack and scroll normally.
 */
export default function Layout({ children, tab, onTabChange, statusSlot = null, headerActions = null }) {
  return (
    <div className="flex min-h-svh flex-col text-fg lg:h-svh lg:overflow-hidden">
      <a href="#main"
        className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-fg focus:not-sr-only focus:absolute focus:left-4 focus:top-3">
        Skip to content
      </a>
      <header className="sticky top-0 z-40 shrink-0 border-b border-line bg-header backdrop-blur-xs">
        <div className="mx-auto flex w-full max-w-[1560px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-ok-line bg-ok-bg" aria-hidden="true">
              <AudioWaveform className="h-5 w-5 text-ok-fg" />
            </div>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold tracking-normal text-fg">{APP_NAME}</h1>
              <p className="text-[11px] font-medium text-fg-subtle">{APP_TAGLINE}</p>
            </div>
          </div>

          <TabList tab={tab} onTabChange={onTabChange} />

          <div className="ml-auto flex items-center gap-3">
            {statusSlot}
            {headerActions}
          </div>
        </div>
      </header>

      <main id="main" tabIndex={-1} className="flex min-h-0 flex-1 flex-col outline-none">
        <div className="mx-auto flex min-h-0 w-full max-w-[1560px] flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </main>
    </div>
  );
}
