import React from 'react';
import { Check, CircleAlert, Cpu, LoaderCircle, RefreshCw } from 'lucide-react';
import { formatBytes } from '../lib/format';

const ASSET_LABELS = {
  hubert: 'HuBERT content encoder',
  rmvpe: 'RMVPE pitch tracker',
  vad: 'Silero voice detector',
};

function AssetRow({ asset }) {
  const label = ASSET_LABELS[asset.id] || asset.file;
  const downloading = asset.state === 'downloading';
  const percent = asset.bytes ? Math.min(100, Math.round((asset.received / asset.bytes) * 100)) : 0;
  let icon = <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn-fg" aria-hidden="true" />;
  let state = asset.required ? 'Missing' : 'Optional, missing';
  if (asset.state === 'ready') {
    icon = <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ok-fg" aria-hidden="true" />;
    state = 'Ready';
  } else if (downloading) {
    icon = <LoaderCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-fg-subtle" aria-hidden="true" />;
    state = `Downloading ${percent}%`;
  } else if (asset.state === 'error') {
    state = 'Download failed';
  }
  return (
    <li className="flex items-start gap-2 py-2 text-xs">
      {icon}
      <div className="min-w-0 flex-1">
        <span className="text-fg-secondary">{label}</span>
        <span className="ml-2 font-mono text-[11px] text-fg-faint">{formatBytes(asset.bytes)}</span>
        {asset.error ? <span className="mt-1 block break-words text-[11px] text-danger-fg">{asset.error}</span> : null}
        {downloading ? (
          <span className="mt-1.5 block h-1 overflow-hidden rounded-full bg-meter-track" aria-hidden="true">
            <span className="block h-full bg-ok-solid" style={{ width: `${percent}%` }} />
          </span>
        ) : null}
      </div>
      <span className="shrink-0 text-fg-subtle">{state}</span>
    </li>
  );
}

/** Inference device and model-asset status (HuBERT is required, RMVPE recommended). */
export default function RuntimeReadiness({ config, checking, error, onRefresh }) {
  const assets = config?.assets || [];
  const device = config?.device;
  const required = assets.filter((asset) => asset.required);
  const ready = required.length > 0 && required.every((asset) => asset.state === 'ready');
  const summary = error
    || (checking && !assets.length ? 'Checking runtime...'
      : ready ? 'Ready. Load a voice in the Models tab to convert.'
        : assets.some((asset) => asset.state === 'downloading')
          ? 'Downloading model assets (first run only). DSP effects work meanwhile.'
          : 'Model assets are missing. Check the network connection, then retry.');

  return (
    <section className="min-w-0 py-4" aria-label="Voice conversion readiness">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-fg-secondary">Voice conversion</h3>
        <button type="button" className="frost-control inline-flex h-9 w-9 shrink-0 items-center justify-center"
          title="Recheck runtime" aria-label="Recheck runtime" onClick={onRefresh} disabled={checking}>
          <RefreshCw className={`h-4 w-4 ${checking ? 'animate-spin' : ''}`} aria-hidden="true" />
        </button>
      </div>
      <p role="status" className={`mt-2 text-xs ${error ? 'text-danger-fg' : 'text-fg-muted'}`}>{summary}</p>
      {device ? (
        <p className="mt-3 flex items-center gap-2 text-xs text-fg-secondary">
          <Cpu className="h-3.5 w-3.5 shrink-0 text-fg-subtle" aria-hidden="true" />
          {device.name}
          <span className="font-mono text-[11px] text-fg-faint">
            {device.device} · {device.half_precision ? 'FP16' : 'FP32'}
            {device.memory_gb ? ` · ${device.memory_gb} GB` : ''}
          </span>
        </p>
      ) : null}
      {assets.length ? <ul className="mt-2 divide-y divide-line border-t border-line">{assets.map((asset) => (
        <AssetRow key={asset.id} asset={asset} />
      ))}</ul> : null}
    </section>
  );
}
