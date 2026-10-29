import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronLeft, ChevronRight, CircleStop, Mic, RotateCcw, Trash2, Upload, UserRound,
} from 'lucide-react';
import {
  deleteVoiceProfile, deleteVoiceReference, fetchVoiceProfile, saveVoiceProfile, saveVoiceReference,
} from '../lib/api';
import { rms } from '../lib/meters';
import { encodeWav } from '../lib/wav';
import {
  PROFILE_LANGUAGES, PROFILE_PROMPTS, PROFILE_TARGET_SECONDS, calibrationSummary, formatShift, registerLabel,
} from '../lib/voiceProfile';
import Button from './Button';
import Picker from './Picker';
import Segmented from './Segmented';

const WORKLET_URL = `${import.meta.env.BASE_URL}audioWorklet.js`;
const CHUNK_SIZE = 2048;
const REFERENCE_ACCEPT = '.wav,.flac,.ogg,.mp3,.m4a,.aiff';

/** Microphone recorder: raw audio (no browser noise suppression or gain riding, which bend pitch and level). */
function useReadingRecorder() {
  const [state, setState] = useState({ recording: false, seconds: 0, level: 0 });
  const sessionRef = useRef(null);

  const stop = useCallback(async () => {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (!session) return null;
    session.node.port.onmessage = null;
    session.stream.getTracks().forEach((track) => track.stop());
    await session.context.close().catch(() => {});
    setState({ recording: false, seconds: 0, level: 0 });
    return { chunks: session.chunks, sampleRate: session.context.sampleRate };
  }, []);

  const start = useCallback(async (deviceId) => {
    await stop();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1,
      },
    });
    const context = new AudioContext();
    try {
      await context.audioWorklet.addModule(WORKLET_URL);
    } catch (err) {
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
      throw err;
    }
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, 'capture-processor', { processorOptions: { chunkSize: CHUNK_SIZE } });
    const sink = context.createGain();
    sink.gain.value = 0;   // the worklet must be pulled, but the microphone is never played
    source.connect(node).connect(sink).connect(context.destination);
    const session = { stream, context, node, chunks: [], samples: 0 };
    node.port.onmessage = ({ data }) => {
      if (!(data instanceof Float32Array)) return;
      session.chunks.push(data);
      session.samples += data.length;
      setState({ recording: true, seconds: session.samples / context.sampleRate, level: rms(data) });
    };
    sessionRef.current = session;
    setState({ recording: true, seconds: 0, level: 0 });
  }, [stop]);

  useEffect(() => () => { stop(); }, [stop]);
  return { ...state, start, stop };
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-line bg-raised px-3 py-2">
      <p className="text-[11px] text-fg-subtle">{label}</p>
      <p className="mt-0.5 font-mono text-sm text-fg">{value}</p>
    </div>
  );
}

function LevelBar({ level }) {
  const db = 20 * Math.log10(Math.max(level, 1e-5));
  const percent = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  const tone = db > -3 ? 'bg-danger-fg' : db > -40 ? 'bg-ok-fg' : 'bg-fg-faint';
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-sunken" role="meter" aria-label="Microphone level"
      aria-valuemin={-60} aria-valuemax={0} aria-valuenow={Math.round(db)}>
      <div className={`h-full ${tone} transition-[width] duration-75`} style={{ width: `${percent}%` }} />
    </div>
  );
}

/** "My voice": record the speaker's profile once; every voice calibrates to it. */
function VoiceProfile({ active, activeModel, inputDevices = [] }) {
  const [language, setLanguage] = useState('es');
  const [deviceId, setDeviceId] = useState('');
  const [line, setLine] = useState(0);
  const [summary, setSummary] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const recorder = useReadingRecorder();
  const referenceInputRef = useRef(null);
  const prompts = PROFILE_PROMPTS[language];

  const refresh = useCallback(async () => {
    try {
      setSummary(await fetchVoiceProfile());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => { if (active) refresh(); }, [active, activeModel, refresh]);

  const run = useCallback(async (label, action) => {
    setBusy(label);
    setError(null);
    try {
      const result = await action();
      if (result) setSummary(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  }, []);

  const startReading = () => run('start', async () => {
    setLine(0);
    await recorder.start(deviceId || undefined);
  });

  const finishReading = () => run('analyse', async () => {
    const take = await recorder.stop();
    if (!take?.chunks.length) return null;
    return saveVoiceProfile(encodeWav(take.chunks, take.sampleRate), language);
  });

  const cancelReading = () => { recorder.stop(); setLine(0); };

  const profile = summary?.profile;
  const voice = summary?.voice;
  const progress = Math.min(100, (recorder.seconds / PROFILE_TARGET_SECONDS) * 100);

  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-5 lg:col-span-2" aria-labelledby="profile-record-title">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="panel-kicker text-ok-fg">Voice</p>
            <h2 id="profile-record-title" className="panel-title">Read a few lines</h2>
          </div>
          <UserRound className="hidden h-7 w-7 shrink-0 text-fg-subtle sm:block" strokeWidth={1.5} aria-hidden="true" />
        </div>
        <div className="mt-4 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          <p className="text-xs leading-5 text-fg-secondary">
            About a minute and a half, once. Your pitch, range and level are measured from it, and every voice
            (trained or downloaded) is tuned to you automatically when you activate it.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Segmented label="Language" accent="voice" value={language} disabled={recorder.recording}
              onChange={(value) => { setLanguage(value); setLine(0); }}
              options={PROFILE_LANGUAGES.map(({ id, label }) => ({ value: id, label }))} />
            <div>
              <p className="mb-1.5 text-xs font-medium text-fg-muted">Microphone</p>
              <Picker label="Microphone" icon={Mic} accent="voice" value={deviceId} disabled={recorder.recording}
                onChange={setDeviceId}
                options={[{ value: '', label: 'System default' },
                  ...inputDevices.map((device) => ({ value: device.deviceId, label: device.label || 'Microphone' }))]} />
            </div>
          </div>

          <div className="rounded-lg border border-line-strong bg-raised px-4 py-4" aria-live="polite">
            <div className="flex items-center justify-between gap-3 text-[11px] text-fg-subtle">
              <span>{prompts[line].kind}</span>
              <span className="font-mono">{line + 1} / {prompts.length}</span>
            </div>
            <p className={`mt-2 text-base leading-7 ${recorder.recording ? 'text-fg' : 'text-fg-muted'}`}>
              {prompts[line].text}
            </p>
          </div>

          {recorder.recording ? (
            <div className="space-y-2">
              <LevelBar level={recorder.level} />
              <div className="h-1 w-full overflow-hidden rounded-full bg-sunken">
                <div className="h-full bg-primary" style={{ width: `${progress}%` }} />
              </div>
              <p className="font-mono text-[11px] text-fg-subtle">
                {Math.floor(recorder.seconds)} s of about {PROFILE_TARGET_SECONDS} s
              </p>
            </div>
          ) : null}

          <div className="flex flex-wrap gap-2">
            {recorder.recording ? (
              <>
                <Button icon={ChevronLeft} onClick={() => setLine((value) => Math.max(0, value - 1))}
                  disabled={line === 0} aria-label="Previous line" />
                {line < prompts.length - 1 ? (
                  <Button variant="primary" icon={ChevronRight} onClick={() => setLine((value) => value + 1)}>
                    Next line
                  </Button>
                ) : (
                  <Button variant="primary" icon={CircleStop} onClick={finishReading} loading={busy === 'analyse'}>
                    Finish and analyse
                  </Button>
                )}
                <Button onClick={cancelReading} disabled={busy === 'analyse'}>Cancel</Button>
              </>
            ) : (
              <Button variant="primary" icon={Mic} onClick={startReading} loading={busy === 'start' || busy === 'analyse'}>
                {profile ? 'Record again' : 'Start reading'}
              </Button>
            )}
          </div>
          {error ? <p className="text-xs text-danger-fg" role="alert">{error}</p> : null}
        </div>
      </section>

      <section className="panel flex min-h-0 flex-col p-5" aria-labelledby="profile-result-title">
        <div className="min-w-0">
          <p className="panel-kicker text-ok-fg">Voice</p>
          <h2 id="profile-result-title" className="panel-title">{profile ? 'Your voice' : 'No profile yet'}</h2>
        </div>
        <div className="mt-4 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
          {profile ? (
            <>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                <Stat label="Typical pitch" value={`${Math.round(profile.median_hz)} Hz`} />
                <Stat label="Register" value={registerLabel(profile.register)} />
                <Stat label="Range" value={`${profile.range_semitones} st`} />
                <Stat label="Voice heard" value={`${Math.round(profile.voiced_seconds)} s`} />
                <Stat label="Speech level" value={`${profile.speech_level_db} dB`} />
                <Stat label="Mic noise" value={`${profile.noise_floor_db} dB`} />
              </div>
              {profile.warnings?.length ? (
                <ul className="space-y-1 text-xs text-warn-fg">
                  {profile.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              ) : null}
            </>
          ) : (
            <p className="text-xs leading-5 text-fg-secondary">
              Until you record one, the studio learns your pitch from the first seconds you speak, every session.
            </p>
          )}

          <div className="rounded-lg border border-line bg-raised px-4 py-3">
            <p className="text-xs font-medium text-fg-muted">Active voice</p>
            <p className="mt-1 text-sm text-fg">{calibrationSummary(voice, profile)}</p>
            {voice?.target ? (
              <p className="mt-1 font-mono text-[11px] text-fg-subtle">
                Voice register {Math.round(voice.target.median_hz)} Hz
                {voice.register_source === 'reference' ? ' (from your clip)' : ' (from its training)'}
                {profile && voice.calibration ? ` · shift ${formatShift(voice.calibration.pitch_shift)}` : ''}
              </p>
            ) : null}
            {voice?.uses_pitch && voice.register_source !== 'training' ? (
              <div className="mt-3 flex flex-wrap gap-2">
                <input ref={referenceInputRef} type="file" accept={REFERENCE_ACCEPT} className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = '';
                    if (file) run('reference', () => saveVoiceReference(voice.name, file));
                  }} />
                <Button icon={Upload} onClick={() => referenceInputRef.current?.click()} loading={busy === 'reference'}>
                  {voice.register_source === 'reference' ? 'Replace original clip' : 'Add a clip of the original voice'}
                </Button>
                {voice.register_source === 'reference' ? (
                  <Button icon={RotateCcw} onClick={() => run('reference', () => deleteVoiceReference(voice.name))}>
                    Forget clip
                  </Button>
                ) : null}
              </div>
            ) : null}
            {voice?.uses_pitch && voice.register_source !== 'training' ? (
              <p className="mt-2 text-[11px] leading-4 text-fg-faint">
                10–60 s of that voice speaking alone (no music) tells the studio its register.
              </p>
            ) : null}
          </div>

          {profile ? (
            <div>
              <Button variant="danger" icon={Trash2} loading={busy === 'delete'}
                onClick={() => run('delete', deleteVoiceProfile)}>
                Delete profile
              </Button>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export default VoiceProfile;
