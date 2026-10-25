// @vitest-environment jsdom
import React from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import LogsPanel from './LogsPanel';

const lines = Array.from({ length: 3000 }, (_, index) => ({
  id: index + 1, time: 1_700_000_000 + index, level: 'INFO', logger: 'backend', message: `line ${index + 1}`,
}));

vi.mock('../lib/api', () => ({
  fetchLogs: vi.fn(() => Promise.resolve({ lines, last_id: lines.length })),
  fetchTrainingJobs: vi.fn(() => Promise.resolve([])),
  fetchTrainingLog: vi.fn(() => Promise.resolve({ lines: [] })),
}));

class FakeSocket {
  close() {}
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('keeps a fixed-size box that scrolls inside instead of growing the page', async () => {
  vi.stubGlobal('WebSocket', FakeSocket);
  render(React.createElement(LogsPanel, { active: true }));
  await waitFor(() => expect(screen.getByText('line 3000')).toBeTruthy());
  const box = screen.getByRole('log');
  expect(box.className).toContain('overflow-y-auto');
  expect(box.className).toContain('h-full');
  expect(box.parentElement.className).toContain('min-h-0');
  // The panel fills the screen below the header (flex-1), never more.
  expect(box.closest('section').className).toContain('lg:flex-1');
});
