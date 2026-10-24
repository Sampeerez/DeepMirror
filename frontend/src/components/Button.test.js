// @vitest-environment jsdom
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { Download } from 'lucide-react';
import Button from './Button';

const h = React.createElement;

function classesOf(element) {
  const html = renderToStaticMarkup(element);
  return html.match(/class="([^"]*)"/)[1].split(' ');
}

describe('Button', () => {
  it('keeps one height per size whatever the variant, icon or state', () => {
    const variants = ['primary', 'secondary', 'ghost', 'danger', 'success', 'warning'];
    for (const variant of variants) {
      expect(classesOf(h(Button, { variant }, 'Activate'))).toContain('h-8');
      expect(classesOf(h(Button, { variant, loading: true }, 'Loading…'))).toContain('h-8');
      expect(classesOf(h(Button, { variant, size: 'lg', icon: Download }, 'Start'))).toContain('h-11');
    }
    expect(classesOf(h(Button, { icon: Download, 'aria-label': 'Download' }))).toEqual(
      expect.arrayContaining(['h-8', 'w-8']),
    );
  });

  it('keeps the width it is given when the label changes', () => {
    const idle = classesOf(h(Button, { className: 'w-28' }, 'Activate'));
    const busy = classesOf(h(Button, { className: 'w-28', loading: true }, 'Loading…'));
    const active = classesOf(h(Button, { className: 'w-28', variant: 'success' }, 'Active'));
    for (const classes of [idle, busy, active]) expect(classes).toContain('w-28');
  });

  it('is disabled and announces itself busy while loading', () => {
    const html = renderToStaticMarkup(h(Button, { loading: true }, 'Saving'));
    expect(html).toContain('disabled');
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('animate-spin');
  });

  it('renders a link that looks the same', () => {
    const html = renderToStaticMarkup(h(Button, { href: '/file.zip', download: true }, 'Download'));
    expect(html.startsWith('<a')).toBe(true);
    expect(html).toContain('href="/file.zip"');
    expect(html).toContain('h-8');
  });
});
