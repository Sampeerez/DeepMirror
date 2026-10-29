// Tab <-> URL synchronisation.
//
// One flat list of pages, shown in the sidebar. Every page is a tab id in
// ?tab=, so links and Back/Forward work page by page. Ids from earlier
// layouts still open the page that now holds their content.

export const PAGE_IDS = ['studio', 'models', 'faces', 'training', 'library', 'settings'];
// The sidebar keeps Settings apart, at the bottom.
export const MAIN_PAGES = PAGE_IDS.filter((id) => id !== 'settings');
export const TAB_IDS = PAGE_IDS;
const DEFAULT_TAB = 'studio';

const MOVED = {
  voice: 'training',
  'voice-settings': 'settings',
  'face-settings': 'settings',
  converter: 'library',
  files: 'library',
};

export function isValidTab(tab) {
  return PAGE_IDS.includes(tab);
}

/** Read the tab from a location search string (?settings opens Settings), falling back to the default. */
export function readTabFromSearch(search) {
  const params = new URLSearchParams(search || '');
  const param = params.get('tab');
  const tab = MOVED[param] || param;
  if (isValidTab(tab)) return tab;
  return params.has('settings') ? 'settings' : DEFAULT_TAB;
}

/**
 * URL for a tab, relative to the current one.
 *
 * The default tab drops ?tab= entirely so the base URL stays clean; ?settings
 * is dropped (the tab says it now), anything else plus the hash survive.
 */
export function buildTabUrl(currentUrl, tab) {
  const url = new URL(currentUrl);
  url.searchParams.delete('settings');
  if (tab === DEFAULT_TAB) {
    url.searchParams.delete('tab');
  } else {
    url.searchParams.set('tab', tab);
  }
  return `${url.pathname}${url.search}${url.hash}`;
}
