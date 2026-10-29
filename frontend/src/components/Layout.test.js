import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import Layout from './Layout';
import { TAB_IDS, ZONES } from '../lib/navigation';
import { APP_NAME } from '../lib/brand';

function render(tab) {
  return renderToStaticMarkup(React.createElement(Layout, { tab }, 'content'));
}

describe('Layout tab list', () => {
  it('shows one tab per zone, and the pages of the zone in use', () => {
    expect(render('studio').match(/role="tab"/g)).toHaveLength(ZONES.length);
    const voice = render('voice');
    expect(voice).toContain('role="tablist"');
    expect(voice.match(/role="tab"/g)).toHaveLength(ZONES.length - 1 + 4);   // Voices, My voice, Train, Settings
    for (const id of ['models', 'voice', 'training', 'voice-settings']) expect(voice).toContain(`id="tab-${id}"`);
    expect(render('faces')).toContain('id="tab-face-settings"');
  });

  it('points each tab at the panel App renders for it', () => {
    for (const tab of TAB_IDS) {
      const html = render(tab);
      expect(html).toContain(`id="tab-${tab}"`);
      expect(html).toContain(`aria-controls="panel-${tab}"`);
    }
  });

  it('keeps only the selected tab in the Tab order', () => {
    const html = render('converter');
    expect(html).toContain('aria-selected="true"');
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1);
    expect(html.match(/tabindex="0"/g)).toHaveLength(1);
    expect(html.match(/role="tab"[^>]*tabindex="-1"/g)).toHaveLength(ZONES.length - 1);
    // The selected one is the converter.
    expect(html).toMatch(/id="tab-converter"[^>]*aria-selected="true"/);
  });

  it('gives the page a top-level heading', () => {
    expect(render('studio')).toContain(`>${APP_NAME}</h1>`);
    expect(render('studio')).toContain('Skip to content');
  });
});
