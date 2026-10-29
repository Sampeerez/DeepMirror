import React, { useRef } from 'react';
import { Boxes, GraduationCap, Library, Radio, ScanFace, Settings2 } from 'lucide-react';
import { APP_NAME } from '../lib/brand';
import { MAIN_PAGES } from '../lib/navigation';
import logoUrl from '../assets/logo.svg';

// Page ids match lib/navigation; the matching panels in App carry
// id="panel-<id>" and aria-labelledby="tab-<id>".
const PAGES = {
  studio: { label: 'Live', Icon: Radio },
  models: { label: 'Voices', Icon: Boxes },
  faces: { label: 'Faces', Icon: ScanFace },
  training: { label: 'Train', Icon: GraduationCap },
  library: { label: 'Library', Icon: Library },
  settings: { label: 'Settings', Icon: Settings2 },
};

function NavItem({ id, selected, onSelect }) {
  const { label, Icon } = PAGES[id];
  return (
    <button
      type="button"
      role="tab"
      id={`tab-${id}`}
      aria-selected={selected}
      aria-controls={`panel-${id}`}
      tabIndex={selected ? 0 : -1}
      data-tab={id}
      title={label}
      onClick={() => onSelect(id)}
      className={`group relative flex h-10 shrink-0 items-center gap-3 rounded-md px-3 text-sm font-medium transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
        selected ? 'bg-brand-bg text-fg' : 'text-fg-muted hover:bg-control-hover hover:text-fg'}`}
    >
      <span aria-hidden="true" className={`absolute inset-y-2 left-0 w-[3px] rounded-full ${selected ? 'bg-primary' : 'bg-transparent'}`} />
      <Icon className={`h-[18px] w-[18px] shrink-0 ${selected ? 'text-brand-fg' : ''}`} aria-hidden="true" />
      <span className="max-lg:sr-only">{label}</span>
    </button>
  );
}

/**
 * The sidebar: every page always in view, Settings at the bottom with the
 * connection status. One tablist (WAI-ARIA tabs, vertical): the arrow keys
 * move across it, only the selected page sits in the Tab order. Below lg it
 * folds into a row of icons along the top.
 */
function Sidebar({ tab, onTabChange, statusSlot }) {
  const listRef = useRef(null);
  const select = (id) => onTabChange?.(id);
  const handleKeyDown = (event) => {
    const keys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const buttons = [...(listRef.current?.querySelectorAll('[role="tab"]') || [])];
    const index = Math.max(0, buttons.findIndex((button) => button.dataset.tab === tab));
    const forward = event.key === 'ArrowDown' || event.key === 'ArrowRight';
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (forward ? 1 : -1) + buttons.length) % buttons.length;
    select(buttons[next].dataset.tab);
    requestAnimationFrame(() => listRef.current?.querySelector(`[data-tab="${buttons[next].dataset.tab}"]`)?.focus());
  };

  return (
    <aside className="z-40 flex shrink-0 items-center gap-3 border-b border-line bg-header px-3 py-2 lg:h-svh lg:w-52 lg:flex-col lg:items-stretch lg:gap-0 lg:border-b-0 lg:border-r lg:px-3 lg:py-4">
      <div className="flex items-center gap-2.5 lg:mb-6 lg:px-1.5">
        <img src={logoUrl} alt="" width={32} height={32} className="h-8 w-8 shrink-0" />
        <h1 className="text-[15px] font-semibold tracking-tight text-fg max-sm:sr-only">{APP_NAME}</h1>
      </div>
      <div
        ref={listRef}
        role="tablist"
        aria-label="Pages"
        aria-orientation="vertical"
        onKeyDown={handleKeyDown}
        className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto lg:flex-col lg:items-stretch"
      >
        {MAIN_PAGES.map((id) => <NavItem key={id} id={id} selected={tab === id} onSelect={select} />)}
        <div className="lg:mt-auto" />
        <NavItem id="settings" selected={tab === 'settings'} onSelect={select} />
      </div>
      {statusSlot ? <div className="shrink-0 lg:mt-3 lg:border-t lg:border-line lg:px-1.5 lg:pt-3">{statusSlot}</div> : null}
    </aside>
  );
}

/**
 * The page frame. On desktop it is exactly one screen tall: the sidebar, then
 * the active page, which fills the rest and scrolls inside its own panels if
 * it must, never the page. Narrow screens stack and scroll normally.
 */
export default function Layout({ children, tab, onTabChange, statusSlot = null }) {
  return (
    <div className="flex min-h-svh flex-col text-fg lg:h-svh lg:flex-row lg:overflow-hidden">
      <a href="#main"
        className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-fg focus:not-sr-only focus:absolute focus:left-4 focus:top-3">
        Skip to content
      </a>
      <Sidebar tab={tab} onTabChange={onTabChange} statusSlot={statusSlot} />
      <main id="main" tabIndex={-1} className="flex min-h-0 min-w-0 flex-1 flex-col outline-none">
        <div className="mx-auto flex min-h-0 w-full max-w-[1600px] flex-1 flex-col px-4 py-4 sm:px-5 lg:px-6">
          {children}
        </div>
      </main>
    </div>
  );
}
