// @vitest-environment jsdom
import React from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import Collapsible from './Collapsible';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('folds to its title and summary, opens on click and remembers it', () => {
  const stored = new Map();
  vi.stubGlobal('localStorage', { getItem: (k) => stored.get(k) ?? null, setItem: (k, v) => stored.set(k, v) });
  const view = () => render(React.createElement(Collapsible, { id: 'fx', title: 'Effects Rack', summary: '2 effects on' },
    React.createElement('p', null, 'inside')));
  view();
  expect(screen.getByText('2 effects on')).toBeTruthy();
  expect(screen.queryByText('inside')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: /Effects Rack/ }));
  expect(screen.getByText('inside')).toBeTruthy();
  cleanup();
  view();
  expect(screen.getByText('inside')).toBeTruthy();   // reopened as it was left
});
