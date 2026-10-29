import React, { useCallback, useEffect, useState } from 'react';
import { Headphones, LoaderCircle, PhoneCall } from 'lucide-react';
import Button from './Button';
import { APP_NAME } from '../lib/brand';
import Toggle from './Toggle';
import { fetchVirtualMic, installVirtualMic } from '../lib/api';
import { findCableOutput } from '../lib/callMode';


/**
 * Call mode: the converted voice goes to a virtual microphone that Discord,
 * Teams or Meet can use, and you do not hear yourself (unless you pick a
 * monitor). The virtual cable is VB-CABLE (Windows, macOS) or a PipeWire /
 * PulseAudio device the studio creates (Linux).
 */
export default function CallMode({ devices, enabled, onEnabledChange, monitorId, onMonitorChange, running, onCallInput }) {
  const [status, setStatus] = useState(null);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await fetchVirtualMic());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (status?.call_input) onCallInput?.(status.call_input); }, [status, onCallInput]);

  const busy = status && ['downloading', 'installing'].includes(status.state);
  useEffect(() => {
    if (!busy) return undefined;
    const timer = setInterval(() => {
      refresh();
      devices.refresh?.().catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, [busy, refresh, devices]);

  const cable = findCableOutput(devices.outputDevices);
  const ready = Boolean(status?.installed && cable);

  const install = async () => {
    setConfirming(false);
    setError(null);
    try {
      setStatus(await installVirtualMic());
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <PhoneCall className="h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-fg-secondary">Send my voice to calls</p>
          </div>
        </div>
        <Toggle checked={enabled && ready} disabled={!ready || running} onChange={onEnabledChange}
          label="Call mode" />
      </div>

      {status && !status.supported ? (
        <p className="mt-3 text-[11px] text-warn-fg">Virtual microphones are not available on this system.</p>
      ) : null}

      {status?.supported && !status.installed && !busy ? (
        <div className="mt-3 rounded-md border border-line bg-raised p-3 text-[11px] leading-4 text-fg-subtle">
          {!confirming ? (
            <>
              <span className="flex items-center justify-between gap-3">
                Virtual microphone not installed
                <Button variant="primary" onClick={() => setConfirming(true)}>Set up</Button>
              </span>
            </>
          ) : status.platform === 'Linux' ? (
            <>
              Creates “{APP_NAME} Microphone” while {APP_NAME} runs.
              <span className="mt-2 flex"><Button variant="primary" onClick={install}>Create it</Button></span>
            </>
          ) : (
            <>
              Installs VB-CABLE (free, by VB-Audio): click <b>Install Driver</b> and accept the prompt.
              <span className="mt-2 flex gap-2">
                <Button variant="primary" onClick={install}>Download and install</Button>
                <Button onClick={() => setConfirming(false)}>Cancel</Button>
              </span>
            </>
          )}
        </div>
      ) : null}

      {busy || status?.message ? (
        <p className={`mt-3 flex items-center gap-2 text-[11px] ${status?.state === 'error' ? 'text-danger-fg' : 'text-fg-subtle'}`}
          role="status">
          {busy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : null}
          {busy && status?.state === 'downloading' ? 'Downloading VB-CABLE…' : status?.message}
        </p>
      ) : null}
      {status?.installed && !cable ? (
        <p className="mt-3 text-[11px] text-warn-fg">
          Installed: refresh the devices{status.platform !== 'Linux' ? ' or restart the computer' : ''}.
        </p>
      ) : null}
      {error ? <p className="mt-3 break-words text-[11px] text-danger-fg">{error}</p> : null}

      {enabled && ready ? (
        <div className="mt-4 space-y-3">
          <div>
            <label htmlFor="call-monitor" className="flex items-center gap-2 text-xs font-medium text-fg-muted">
              <Headphones className="h-3.5 w-3.5" aria-hidden="true" /> Hear yourself
            </label>
            <select id="call-monitor" value={monitorId} disabled={running}
              onChange={(event) => onMonitorChange(event.target.value)}
              className="native-select-safe field mt-2">
              <option value="">No</option>
              {devices.outputDevices.filter((device) => device.deviceId !== cable.deviceId).map((device) => (
                <option key={device.deviceId} value={device.deviceId}>{device.label || 'Speaker'}</option>
              ))}
            </select>
          </div>
          <p className="rounded-md border border-line bg-raised p-3 text-xs text-fg-subtle">
            In the call app, microphone: <b className="text-fg-secondary">{status.call_input}</b>
            {' '}· camera: <b className="text-fg-secondary">OBS Virtual Camera</b>
          </p>
        </div>
      ) : null}
    </div>
  );
}
