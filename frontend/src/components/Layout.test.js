import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import Layout from './Layout';
import { PAGE_IDS } from '../lib/navigation';
import { APP_NAME } from '../lib/brand';

function render(tab) {
  return renderToStaticMarkup(React.createElement(Layout, { tab }, 'content'));
}

describe('Layout sidebar', () => {
  it('shows every page, always', () => {
    for (const tab of PAGE_IDS) expect(render(tab).match(/role="tab"/g)).toHaveLength(PAGE_IDS.length);
    expect(render('studio')).toContain('aria-orientation="vertical"');
  });

  it('points each tab at the panel App renders for it', () => {
    const html = render('studio');
    for (const tab of PAGE_IDS) {
      expect(html).toContain(`id="tab-${tab}"`);
      expect(html).toContain(`aria-controls="panel-${tab}"`);
    }
  });

  it('keeps only the selected tab in the Tab order', () => {
    const html = render('library');
    expect(html.match(/aria-selected="true"/g)).toHaveLength(1);
    expect(html.match(/tabindex="0"/g)).toHaveLength(1);
    expect(html).toMatch(/id="tab-library"[^>]*aria-selected="true"/);
  });

  it('gives the page a top-level heading', () => {
    expect(render('studio')).toContain(`>${APP_NAME}</h1>`);
    expect(render('studio')).toContain('Skip to content');
  });
});
