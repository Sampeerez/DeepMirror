import React from 'react';

const STATUS_CONFIG = {
  connected: { color: 'bg-ok-solid', label: 'Connected' },
  connecting: { color: 'bg-warn-solid', label: 'Connecting' },
  disconnected: { color: 'bg-fg-subtle', label: 'Offline' },
  error: { color: 'bg-danger-solid', label: 'Error' },
};

// Compact header status: connection dot, engine mode, active model.
export default function StatusIndicator({ wsStatus, activeModel, mode, bypass = false }) {
  const failed = mode === 'error' && !bypass;
  const config = failed && wsStatus === 'connected'
    ? { color: 'bg-danger-solid', label: 'Muted' }
    : STATUS_CONFIG[wsStatus] || STATUS_CONFIG.disconnected;
  // Bypass replaces the engine label: while it is on, neither the model nor
  // the DSP chain is doing anything, so naming either one would be a lie.
  const bypassed = bypass || mode === 'bypass';
  const engineLabel = bypassed
    ? 'BYPASS'
    : failed ? 'ERROR' : activeModel
      ? 'RVC'
      : 'DSP';

  const detail = bypassed ? 'Original voice' : failed ? 'Model error: muted' : (activeModel || 'Effects only');
  return (
    <div className="flex min-w-0 items-center gap-2.5" title={`${engineLabel} · ${detail}`}>
      <span className="relative flex h-2 w-2 shrink-0">
        <span className={`h-2 w-2 rounded-full ${config.color}`} />
        {wsStatus === 'connecting' && (
          <span className={`absolute inset-0 h-2 w-2 rounded-full ${config.color} animate-ping opacity-75`} />
        )}
      </span>
      <span className="min-w-0 max-lg:hidden">
        <span className="block text-xs font-medium text-fg-secondary">{config.label}</span>
        <span className={`block truncate text-[11px] ${bypassed ? 'text-warn-fg' : 'text-fg-subtle'}`}>
          {detail.replace(/\.(pth|pt)$/i, '')}
        </span>
      </span>
    </div>
  );
}
