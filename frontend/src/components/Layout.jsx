import React, { useEffect, useRef, useState } from 'react';
import {
  AudioLines, Boxes, Files, GraduationCap, Radio, ScanFace, Settings2, SlidersHorizontal, UserRound,
} from 'lucide-react';
import { APP_NAME, APP_TAGLINE } from '../lib/brand';
import { ZONES, zoneOf } from '../lib/navigation';
import logoUrl from '../assets/logo.svg';

// Page ids match lib/navigation; the matching panels in App carry
// id="panel-<id>" and aria-labelledby="tab-<id>".
const PAGES = {
  studio: { label: 'Live', Icon: Radio },
  models: { label: 'Voices', Icon: Boxes },
  voice: { label: 'My voice', Icon: UserRound },
  training: { label: 'Train', Icon: GraduationCap },
  'voice-settings': { label: 'Settings', Icon: Settings2 },
  faces: { label: 'Faces', Icon: Boxes },
  'face-settings': { label: 'Settings', Icon: SlidersHorizontal },
  converter: { label: 'Files', Icon: Files },
};
const ZONE_ICONS = { voice: AudioLines, face: ScanFace };
// The voice and the face each have their colour, so it is always clear which side a page belongs to.
const ZONE_ACCENT = {
  voice: { text: 'text-ok-fg', group: 'border-ok-line bg-ok-bg', chip: 'text-ok-fg' },
  face: { text: 'text-video-fg', group: 'border-video-line bg-video-bg', chip: 'text-video-fg' },
};

function TabButton({ id, label, Icon, selected, onSelect, accent = '' }) {
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${id}`}
      aria-selected={selected}
      aria-controls={`panel-${id}`}
      tabIndex={selected ? 0 : -1}
      data-tab={id}
      data-selected={selected}
      onClick={() => onSelect(id)}
      className={`flex h-8 min-w-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md border px-2.5 text-xs font-medium sm:gap-2 sm:px-3 ${
        selected
          ? `frost-control ${accent || 'text-fg'}`
          : 'border-transparent text-fg-muted transition-colors duration-150 ease-out hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]'
      }`}
    >
      <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
      {label}
    </button>
  );
}

/**
 * Top-level navigation: Live, Voice, Face, Files. The zone in use opens in
 * place to show its own pages (Voice: Voices, My voice, Train, Settings;
 * Face: Faces, Settings), in the zone's colour.
 *
 * One tablist: arrow keys move across every visible tab and only the
 * selected one sits in the Tab order (WAI-ARIA tabs pattern).
 */
function TabList({ tab, onTabChange }) {
  const listRef = useRef(null);
  const current = zoneOf(tab);
  // A zone reopens on the page last used in it.
  const [lastInZone, setLastInZone] = useState({});
  useEffect(() => {
    setLastInZone((previous) => (previous[current.id] === tab ? previous : { ...previous, [current.id]: tab }));
  }, [current.id, tab]);

  const select = (id) => onTabChange?.(id);
  const handleKeyDown = (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...(listRef.current?.querySelectorAll('[role="tab"]') || [])];
    const index = Math.max(0, buttons.findIndex((button) => button.dataset.selected === 'true'));
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    select(buttons[next].dataset.tab);
    requestAnimationFrame(() => listRef.current?.querySelector(`[data-tab="${buttons[next].dataset.tab}"]`)?.focus());
  };

  return (
    <div
      ref={listRef}
      role="tablist"
      aria-label="Workspace"
      onKeyDown={handleKeyDown}
      className="order-3 flex w-full flex-wrap items-center gap-1 rounded-lg border border-[color:var(--frost-border)] bg-sunken p-1 sm:order-none sm:w-auto"
    >
      {ZONES.map((zone) => {
        const accent = ZONE_ACCENT[zone.id];
        if (zone.id === current.id && zone.tabs.length > 1) {
          const ZoneIcon = ZONE_ICONS[zone.id];
          return (
            <div key={zone.id} className={`flex items-center gap-0.5 rounded-md border py-0.5 pl-2.5 pr-0.5 ${accent.group}`}>
              <span className={`mr-1 flex items-center gap-1.5 text-xs font-semibold ${accent.chip}`}>
                <ZoneIcon className="h-4 w-4" aria-hidden="true" />
                {zone.label}
              </span>
              {zone.tabs.map((id) => (
                <TabButton key={id} id={id} {...PAGES[id]} selected={tab === id} onSelect={select}
                  accent={accent.text} />
              ))}
            </div>
          );
        }
        const target = lastInZone[zone.id] || zone.tabs[0];
        const page = zone.tabs.length > 1 ? { label: zone.label, Icon: ZONE_ICONS[zone.id] } : PAGES[target];
        return <TabButton key={zone.id} id={target} {...page} selected={tab === target} onSelect={select} />;
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
      <header className="sticky top-0 z-40 shrink-0 border-b border-line bg-header">
        <div className="mx-auto flex w-full max-w-[1560px] flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <img src={logoUrl} alt="" width={36} height={36} className="h-9 w-9 shrink-0" />
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
