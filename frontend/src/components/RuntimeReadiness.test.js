import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import RuntimeReadiness from './RuntimeReadiness';
import StatusIndicator from './StatusIndicator';

const asset = (id, state, extra = {}) => ({
  id, file: `${id}.pt`, required: id === 'hubert', state, bytes: 1000, received: 0, error: null, ...extra,
});

it('reports readiness and the inference device once the encoder is present', () => {
  const html = renderToStaticMarkup(React.createElement(RuntimeReadiness, {
    config: {
      assets: [asset('hubert', 'ready'), asset('rmvpe', 'ready')],
      device: { name: 'GeForce', device: 'cuda:0', half_precision: false, memory_gb: 6 },
    },
    onRefresh: () => {},
  }));
  expect(html).toContain('Ready. Activate a voice');
  expect(html).toContain('cuda:0 · FP32 · 6 GB');
  expect(html).toContain('Recheck runtime');
});

it('shows first-run download progress and failures', () => {
  const html = renderToStaticMarkup(React.createElement(RuntimeReadiness, {
    config: { assets: [asset('hubert', 'downloading', { received: 250 }), asset('rmvpe', 'error', { error: 'URLError' })] },
    onRefresh: () => {},
  }));
  expect(html).toContain('Downloading 25%');
  expect(html).toContain('Download failed');
  expect(html).toContain('URLError');
});

it('reports muted inference failures in the header, not a healthy RVC mode', () => {
  const html = renderToStaticMarkup(React.createElement(StatusIndicator, {
    wsStatus: 'connected', activeModel: 'model.pth', mode: 'error',
  }));
  expect(html).toContain('Muted');
  expect(html).toContain('ERROR');
  expect(html).not.toContain('>RVC<');
});
