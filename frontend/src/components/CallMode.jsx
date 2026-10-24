import React, { useCallback, useEffect, useState } from 'react';
import { Headphones, LoaderCircle, PhoneCall } from 'lucide-react';
import Button from './Button';
import Toggle from './Toggle';
import { fetchVirtualMic, installVirtualMic } from '../lib/api';
import { findCableOutput } from '../lib/callMode';

const APPS = [
  ['Discord', 'User Settings → Voice & Video → Input Device'],
  ['Teams', 'Settings → Devices → Microphone'],
  ['Google Meet', 'Settings (⋮) → Audio → Microphone'],
];

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
            <p className="text-xs font-medium text-fg-muted">Call mode (Discord, Teams, Meet…)</p>
            <p className="mt-1 break-words text-[11px] text-fg-faint">
              {enabled && ready ? `Your converted voice goes to “${status.call_input}”.`
                : 'Send the converted voice to a virtual microphone instead of your speakers.'}
            </p>
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
              A virtual microphone is needed for calls and is not installed yet.
              <span className="mt-2 flex"><Button onClick={() => setConfirming(true)}>Install virtual microphone</Button></span>
            </>
          ) : status.platform === 'Linux' ? (
            <>
              The studio will create “OpenVoiceChanger Microphone” with PipeWire/PulseAudio (nothing is installed;
              it disappears when the studio closes).
              <span className="mt-2 flex"><Button variant="primary" onClick={install}>Create it</Button></span>
            </>
          ) : (
            <>
              The studio will download VB-CABLE, a free virtual audio cable by VB-Audio (donationware,
              vb-cable.com), and open its installer. Click <b>Install Driver</b> there and accept the administrator
              prompt. VB-Audio recommends restarting the computer afterwards.
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
          The virtual cable is installed but the browser does not list it yet: press the refresh button
          {status.platform !== 'Linux' ? ' (or restart the computer, as VB-Audio recommends)' : ''}.
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
              <option value="">No (recommended in calls)</option>
              {devices.outputDevices.filter((device) => device.deviceId !== cable.deviceId).map((device) => (
                <option key={device.deviceId} value={device.deviceId}>{device.label || 'Speaker'}</option>
              ))}
            </select>
          </div>
          <div className="rounded-md border border-line bg-raised p-3 text-[11px] leading-4 text-fg-subtle">
            In your call app choose <b>{status.call_input}</b> as the microphone:
            <ul className="mt-1 list-disc pl-4">
              {APPS.map(([app, where]) => <li key={app}><b>{app}</b>: {where}</li>)}
            </ul>
            Use headphones, so the other people&apos;s voices do not reach your microphone. Turn off the call
            app&apos;s own noise suppression if your voice sounds muffled.
          </div>
          {running ? <p className="text-[11px] text-fg-faint">Changes apply the next time you start the stream.</p> : null}
        </div>
      ) : null}
    </div>
  );
}
