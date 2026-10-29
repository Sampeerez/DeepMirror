// Tab <-> URL synchronisation, and the two levels of navigation.
//
// The header shows four zones: Live (voice and face together), Voice, Face and
// Files. Voice and Face have pages of their own, each ending in its settings.
// Every page is a tab id in ?tab=, so links and Back/Forward work page by page.

export const ZONES = [
  { id: 'live', label: 'Live', tabs: ['studio'] },
  { id: 'voice', label: 'Voice', tabs: ['models', 'voice', 'training', 'voice-settings'] },
  { id: 'face', label: 'Face', tabs: ['faces', 'face-settings'] },
  { id: 'files', label: 'Files', tabs: ['converter'] },
];

export const TAB_IDS = ZONES.flatMap((zone) => zone.tabs);
const DEFAULT_TAB = 'studio';

export function isValidTab(tab) {
  return TAB_IDS.includes(tab);
}

/** The zone a page belongs to. */
export function zoneOf(tab) {
  return ZONES.find((zone) => zone.tabs.includes(tab)) || ZONES[0];
}

/** Read the tab from a location search string, falling back to the default. */
export function readTabFromSearch(search) {
  const param = new URLSearchParams(search || '').get('tab');
  return isValidTab(param) ? param : DEFAULT_TAB;
}

/**
 * URL for a tab, relative to the current one.
 *
 * The default tab drops ?tab= entirely so the base URL stays clean, and any
 * other query parameter (notably ?settings) plus the hash survive the switch.
 */
export function buildTabUrl(currentUrl, tab) {
  const url = new URL(currentUrl);
  if (tab === DEFAULT_TAB) {
    url.searchParams.delete('tab');
  } else {
    url.searchParams.set('tab', tab);
  }
  return `${url.pathname}${url.search}${url.hash}`;
}
