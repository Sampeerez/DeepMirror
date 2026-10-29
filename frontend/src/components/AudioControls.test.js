// @vitest-environment jsdom
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import AudioControls from './AudioControls';

const devices = {
  inputDevices: [],
  outputDevices: [],
  hasLabels: true,
  permissionState: 'granted',
  error: null,
  captureSupported: true,
  outputSelectionSupported: true,
};

const idlePipeline = { isRunning: false, streamInfo: null };

function render(props) {
  return renderToStaticMarkup(React.createElement(AudioControls, {
    devices,
    pipeline: idlePipeline,
    activeModel: null,
    onRetryConnection: () => {},
    ...props,
  }));
}

describe('AudioControls connection state', () => {
  it('offers an immediate retry while the reconnect back-off is waiting', () => {
    const html = render({ wsStatus: 'disconnected' });
    expect(html).toContain('Server not reachable');
    expect(html).toContain('>Retry<');
  });

  it('keeps the retry available while an attempt hangs in CONNECTING', () => {
    // A dead or unreachable host can leave the socket connecting for seconds;
    // retryNow replaces that attempt, so the button must not disappear.
    const html = render({ wsStatus: 'connecting' });
    expect(html).toContain('Connecting');
    expect(html).toContain('>Retry<');
  });

  it('shows nothing about the connection once it is up', () => {
    const html = render({ wsStatus: 'connected' });
    expect(html).not.toContain('>Retry<');
    expect(html).not.toContain('not reachable');
    expect(html).toContain('Start');
    expect(html).toContain('No voice (effects only)');
  });
});
