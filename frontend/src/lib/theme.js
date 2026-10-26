// Theme: reading, saving and applying it. index.html inlines a copy of normalizeTheme.

export const THEME_STORAGE_KEY = 'ovc_theme';
export const THEMES = ['dark', 'light'];
export const DEFAULT_THEME = 'dark';

/** Coerce anything into a supported theme. Unknown or corrupt input -> dark. */
export function normalizeTheme(value) {
  return THEMES.includes(value) ? value : DEFAULT_THEME;
}

/** The saved theme (a bare string; storage may throw in private mode). */
export function readStoredTheme(storage) {
  try {
    return normalizeTheme(storage?.getItem(THEME_STORAGE_KEY));
  } catch {
    return DEFAULT_THEME;
  }
}

/** Save a theme; returns whether it was stored. */
export function storeTheme(theme, storage) {
  if (typeof storage?.setItem !== 'function') return false;
  try {
    storage.setItem(THEME_STORAGE_KEY, normalizeTheme(theme));
    return true;
  } catch {
    return false;
  }
}

/** Apply a theme: data-theme for the CSS tokens, color-scheme for native widgets. */
export function applyTheme(theme, root) {
  const normalized = normalizeTheme(theme);
  if (!root) return normalized;
  root.setAttribute('data-theme', normalized);
  return normalized;
}

/** The theme currently applied to the document root. */
export function readAppliedTheme(root) {
  return normalizeTheme(root?.getAttribute?.('data-theme'));
}

/** CSS tokens as colour strings for the canvas. Call on theme change, not per frame. */
export function readThemeColors(names, root) {
  const result = {};
  if (!root || typeof window === 'undefined' || !window.getComputedStyle) {
    for (const [key] of Object.entries(names)) result[key] = '';
    return result;
  }
  const styles = window.getComputedStyle(root);
  for (const [key, property] of Object.entries(names)) {
    result[key] = styles.getPropertyValue(property).trim();
  }
  return result;
}
