import React, { memo, useState } from 'react';
import { AudioLines, Boxes, Mic, PhoneCall, Play, RefreshCw, Speaker, Square } from 'lucide-react';
import Button from './Button';
import CallMode from './CallMode';
import Dialog from './Dialog';
import Picker from './Picker';
import { findCableOutput, isCableMicrophone } from '../lib/callMode';
import { readStored, writeStored } from '../lib/storage';

const CALL_MODE_KEY = 'ovc_call_mode_v1';

const TONE_CLASS = { danger: 'text-danger-fg', warn: 'text-warn-fg', muted: 'text-fg-subtle' };

/** The one line under the controls: the most important thing to know right now. */
function statusLine({ error, devices, captureSupported, pipeline, wsStatus, inCall, callInput }) {
  if (error) return { tone: 'danger', text: error };
  if (!captureSupported) return { tone: 'danger', text: devices.error };
  if (devices.permissionState === 'denied') return { tone: 'danger', text: 'Microphone blocked in the browser' };
  if (devices.error) return { tone: 'warn', text: devices.error };
  if (!pipeline.isRunning && wsStatus !== 'connected') {
    return {
      tone: 'muted',
      retry: true,
      text: wsStatus === 'connecting' ? 'Connecting…' : 'Server not reachable',
    };
  }
  const info = pipeline.streamInfo;
  if (info?.inputFallback) {
    return {
      tone: 'warn',
      text: info.inputFallback === 'default'
        ? 'Microphone unavailable: using the default one'
        : info.inputFallback === 'sample-rate'
          ? null
          : 'Using the closest microphone',
    };
  }
  if (pipeline.isRunning && info?.adaptedFrom && info.chunkSize !== info.adaptedFrom) {
    const ms = Math.round((info.chunkSize * 1000) / info.sampleRate);
    return { tone: 'warn', text: `Longer buffer (${ms} ms): the computer is busy` };
  }
  if (pipeline.isRunning && inCall) return { tone: 'muted', text: `In the call as “${callInput}”` };
  return { tone: 'muted', text: null };
}

const NO_VOICE = '';

/**
 * Live › Voice: start/stop, the voice to speak with, microphone, output, call
 * mode and A/B. A status line appears only when there is something to say.
 */
function AudioControls({
  devices, pipeline, wsStatus, activeModel, bypass = false, onBypassChange, onRetryConnection,
  voices = [], loadingModel = null, onVoiceChange = null,
}) {
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
    error, devices, captureSupported, pipeline, wsStatus, inCall, callInput,
  });

  const microphones = [{ value: '', label: 'Default microphone' },
    ...devices.inputDevices.filter((device) => !isCableMicrophone(device)).map((device) => ({
      value: device.deviceId, label: device.label || `Microphone (${device.deviceId.slice(0, 8)}…)`,
    }))];
  // Without setSinkId the browser always plays to the system default: no list to offer.
  const speakers = inCall
    ? [{ value: cable.deviceId, label: cable.label || 'Virtual cable' }]
    : [{ value: '', label: outputSelectable ? 'Default speaker' : 'System default' },
      ...(outputSelectable ? devices.outputDevices.map((device) => ({
        value: device.deviceId, label: device.label || `Speaker (${device.deviceId.slice(0, 8)}…)`,
      })) : [])];
  const voiceOptions = [{ value: NO_VOICE, label: 'No voice (effects only)' },
    ...voices.map((name) => ({ value: name, label: name.replace(/\.(pth|pt)$/i, ''),
      note: name === loadingModel ? 'loading…' : null }))];

  return (
    <section className="panel flex shrink-0 flex-col gap-2.5 p-4" aria-labelledby="live-voice-title">
      <div className="flex h-7 items-center gap-2.5">
        <AudioLines className="h-[18px] w-[18px] shrink-0 text-ok-fg" aria-hidden="true" />
        <h2 id="live-voice-title" className="text-sm font-semibold text-fg">Voice</h2>
        <span className="flex-1" />
        {status.retry && onRetryConnection ? (
          /* The back-off grows to 30 s, and an attempt to an unreachable host can
             hang in CONNECTING; retryNow replaces a pending attempt. */
          <Button icon={RefreshCw} onClick={onRetryConnection}>Retry</Button>
        ) : null}
        <Button icon={RefreshCw} size="sm" variant="ghost" aria-label="Refresh audio devices" title="Refresh devices"
          disabled={pipeline.isRunning || devices.isRefreshing} className={devices.isRefreshing ? '[&_svg]:animate-spin' : ''}
          onClick={() => devices.refresh?.().catch(() => {})} />
      </div>

      {pipeline.isRunning ? (
        <Button variant="danger" size="lg" icon={Square} onClick={handleStop} className="w-full">Stop</Button>
      ) : (
        <Button variant="primary" size="lg" icon={Play} onClick={handleStart} disabled={!canStart}
          loading={starting} className="w-full">
          {starting ? 'Starting…' : 'Start'}
        </Button>
      )}
      {status.text ? (
        <p className={`-mt-0.5 truncate text-xs ${TONE_CLASS[status.tone]}`} role="status" aria-live="polite"
          title={status.text}>{status.text}</p>
      ) : <p className="sr-only" role="status" aria-live="polite">{pipeline.isRunning ? 'Live' : ''}</p>}

      <Picker label="Speak as" icon={Boxes} accent="voice"
        value={activeModel || loadingModel || NO_VOICE} options={voiceOptions}
        onChange={(name) => onVoiceChange?.(name || null)} disabled={!onVoiceChange || Boolean(loadingModel)} />
      <Picker label="Microphone" icon={Mic} accent="voice" value={selectedInput} options={microphones}
        onChange={setSelectedInput} disabled={pipeline.isRunning} />
      <Picker label={inCall ? 'Output (call)' : 'Output'} icon={Speaker} accent="voice"
        value={inCall ? cable.deviceId : selectedOutput} options={speakers} onChange={setSelectedOutput}
        disabled={pipeline.isRunning || !outputSelectable || inCall} />
      <div className="grid grid-cols-2 gap-2">
        <Button icon={PhoneCall} size="md" selected={inCall} onClick={() => setCallOpen(true)}
          className={inCall ? 'text-ok-fg' : ''}>
          {inCall ? 'Call: on' : 'Call'}
        </Button>
        {/* True A/B: keeps the mic, the WebSocket and the output routing live and
            returns the raw input instead of the converted signal. */}
        <Button size="md" variant={bypass ? 'warning' : 'secondary'} role="switch" aria-checked={bypass}
          aria-label="Hear the original (A/B)" title="B"
          onClick={() => onBypassChange?.(!bypass)}>
          {bypass ? 'A/B: original' : 'A/B'}
        </Button>
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
