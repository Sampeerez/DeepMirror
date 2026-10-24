import React, { memo, useState } from 'react';
import { Mic, PhoneCall, Play, RefreshCw, Speaker, Square } from 'lucide-react';
import Button from './Button';
import CallMode from './CallMode';
import Dialog from './Dialog';
import { findCableOutput, isCableMicrophone } from '../lib/callMode';
import { readStored, writeStored } from '../lib/storage';

const CALL_MODE_KEY = 'ovc_call_mode_v1';

const TONE_CLASS = { danger: 'text-danger-fg', warn: 'text-warn-fg', muted: 'text-fg-subtle' };

/** The one line under the controls: the most important thing to know right now. */
function statusLine({ error, devices, captureSupported, pipeline, wsStatus, activeModel, inCall, callInput }) {
  if (error) return { tone: 'danger', text: error };
  if (!captureSupported) return { tone: 'danger', text: devices.error };
  if (devices.permissionState === 'denied') return { tone: 'danger', text: 'Microphone permission is blocked in the browser.' };
  if (devices.error) return { tone: 'warn', text: devices.error };
  if (!pipeline.isRunning && wsStatus !== 'connected') {
    return {
      tone: 'muted',
      retry: true,
      text: wsStatus === 'connecting' ? 'Connecting to the server…' : 'Waiting for server connection — retrying automatically.',
    };
  }
  const info = pipeline.streamInfo;
  if (info?.inputFallback) {
    return {
      tone: 'warn',
      text: info.inputFallback === 'default'
        ? 'The selected microphone was unavailable — recording from the system default instead.'
        : info.inputFallback === 'sample-rate'
          ? 'The microphone rejected the preferred rate — the browser resamples it.'
          : 'The selected microphone could not be matched exactly — the browser picked the closest one.',
    };
  }
  if (info && info.sampleRate !== info.requestedSampleRate) {
    return { tone: 'muted', text: `Streaming at ${info.sampleRate.toLocaleString()} Hz (the browser did not grant ${info.requestedSampleRate.toLocaleString()} Hz).` };
  }
  if (pipeline.isRunning) {
    return {
      tone: 'muted',
      text: inCall ? `Live — your converted voice goes to “${callInput}”. Choose it as the microphone in your call app.`
        : 'Live — speak into your microphone. B switches A/B, R records.',
    };
  }
  if (!activeModel) return { tone: 'muted', text: 'No voice active: effects and pitch only. Load a voice in the Models tab.' };
  if (!devices.hasLabels) return { tone: 'muted', text: 'Press Start and allow the microphone to see device names.' };
  return { tone: 'muted', text: 'Ready. Press Start and speak.' };
}

function DeviceSelect({ id, label, Icon, value, onChange, disabled, title, children }) {
  return (
    <label htmlFor={id} className="flex min-w-0 items-center gap-2" title={title}>
      <Icon className="h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />
      <span className="sr-only">{label}</span>
      <select id={id} value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}
        className="native-select-safe field">
        {children}
      </select>
    </label>
  );
}

/** The studio's top bar: start/stop, microphone, output, call mode and A/B, and a status line. */
function AudioControls({ devices, pipeline, wsStatus, activeModel, bypass = false, onBypassChange, onRetryConnection }) {
  const [selectedInput, setSelectedInput] = useState('');
  const [selectedOutput, setSelectedOutput] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);
  const [callMode, setCallMode] = useState(() => readStored(CALL_MODE_KEY)?.enabled === true);
  const [callOpen, setCallOpen] = useState(false);
  const [callInput, setCallInput] = useState('CABLE Output');
  const [monitorId, setMonitorId] = useState('');
  const cable = findCableOutput(devices.outputDevices);
  const inCall = callMode && Boolean(cable);
  const changeCallMode = (enabled) => {
    setCallMode(enabled);
    writeStored(CALL_MODE_KEY, { enabled });
  };

  const outputSelectable = devices.outputSelectionSupported !== false;
  const captureSupported = devices.captureSupported !== false;
  // A voice model is optional: without one, the stream runs through the
  // server-side DSP chain (pitch + effects) instead.
  const canStart = wsStatus === 'connected' && !pipeline.isRunning && captureSupported;

  const handleStart = async () => {
    if (!canStart) return;
    setError(null);
    setStarting(true);
    try {
      await devices.refresh?.();
      // Call mode: the voice goes to the virtual cable; you hear it only on the chosen monitor.
      const output = inCall ? cable.deviceId : selectedOutput || undefined;
      await pipeline.start(selectedInput || undefined, output, inCall ? monitorId || undefined : undefined);
    } catch (err) {
      setError(err.message || 'Failed to start audio pipeline');
    } finally {
      setStarting(false);
    }
  };

  const handleStop = () => {
    setError(null);
    pipeline.stop();
  };

  const status = statusLine({
    error, devices, captureSupported, pipeline, wsStatus, activeModel, inCall, callInput,
  });

  return (
    <section className="panel p-3 sm:p-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-[10rem_minmax(0,1fr)_minmax(0,1fr)_auto] xl:items-center">
        {pipeline.isRunning ? (
          <Button variant="danger" size="md" icon={Square} onClick={handleStop} className="w-full">Stop</Button>
        ) : (
          <Button variant="primary" size="md" icon={Play} onClick={handleStart} disabled={!canStart}
            loading={starting} className="w-full">
            {starting ? 'Starting…' : 'Start'}
          </Button>
        )}

        <div className="flex items-center gap-2 md:justify-end xl:order-last">
          <Button icon={PhoneCall} size="md" selected={inCall} onClick={() => setCallOpen(true)}
            className={`w-32 ${inCall ? 'text-fg' : ''}`} title="Use the converted voice in Discord, Teams or Meet">
            {inCall ? 'Call: on' : 'Call mode'}
          </Button>
          {/* True A/B: keeps the mic, the WebSocket and the output routing live and
              returns the raw input instead of the converted signal. */}
          <Button size="md" variant={bypass ? 'warning' : 'secondary'} role="switch" aria-checked={bypass}
            aria-label="Conversion bypass" title="Compare with your own voice (shortcut: B)"
            onClick={() => onBypassChange?.(!bypass)} className="w-32">
            {bypass ? 'A/B: original' : 'A/B: converted'}
          </Button>
          <Button icon={RefreshCw} size="md" variant="ghost" aria-label="Refresh audio devices" title="Refresh devices"
            disabled={pipeline.isRunning || devices.isRefreshing} className={devices.isRefreshing ? '[&_svg]:animate-spin' : ''}
            onClick={() => devices.refresh?.().catch(() => {})} />
        </div>

        <DeviceSelect id="audio-input-device" label="Input" Icon={Mic} value={selectedInput}
          onChange={setSelectedInput} disabled={pipeline.isRunning}>
          <option value="">Default microphone</option>
          {devices.inputDevices.filter((device) => !isCableMicrophone(device)).map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Microphone (${device.deviceId.slice(0, 8)}…)`}
            </option>
          ))}
        </DeviceSelect>

        <DeviceSelect id="audio-output-device" label="Output" Icon={Speaker}
          value={inCall ? cable.deviceId : (outputSelectable ? selectedOutput : '')}
          onChange={setSelectedOutput} disabled={pipeline.isRunning || !outputSelectable || inCall}
          title={outputSelectable ? undefined : 'This browser cannot route audio to a specific device: it follows the system default.'}>
          {/* Without setSinkId the browser always plays to the system default,
              so offering a device list here would be a promise we can't keep. */}
          <option value="">{outputSelectable ? 'Default speaker' : 'System default (not selectable)'}</option>
          {outputSelectable && devices.outputDevices.map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Speaker (${device.deviceId.slice(0, 8)}…)`}
            </option>
          ))}
        </DeviceSelect>
      </div>

      <div className="mt-2 flex h-8 items-center justify-between gap-3" role="status">
        <p className={`min-w-0 truncate text-[11px] ${TONE_CLASS[status.tone]}`} title={status.text}>{status.text}</p>
        {status.retry && onRetryConnection ? (
          /* The back-off grows to 30 s, and an attempt to an unreachable host can
             hang in CONNECTING; retryNow replaces a pending attempt. */
          <Button icon={RefreshCw} onClick={onRetryConnection}>Retry now</Button>
        ) : null}
      </div>

      {callOpen ? (
        <Dialog title="Call mode (Discord, Teams, Meet…)" onClose={() => setCallOpen(false)}>
          <CallMode devices={devices} enabled={callMode} onEnabledChange={changeCallMode} monitorId={monitorId}
            onMonitorChange={setMonitorId} running={pipeline.isRunning} onCallInput={setCallInput} />
        </Dialog>
      ) : null}
    </section>
  );
}

export default memo(AudioControls);
