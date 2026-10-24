import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CircleX, Cpu, Download, GraduationCap, LoaderCircle, Play, RotateCcw, Square, Trash2, Upload, Wand2,
} from 'lucide-react';
import {
  activateModel, createTrainingJob, deleteTrainingJob, fetchTrainingInfo, fetchTrainingJobs, modelDownloadUrl,
  resumeTrainingJob, stopTrainingJob,
} from '../lib/api';
import { formatBytes } from '../lib/format';
import {
  STAGE_LABELS, STATE_LABELS, bestEpochSummary, canResume, datasetQuality, formatDuration, isActiveJob, jobModelFile,
  plannedEpochs, readAudioDuration,
} from '../lib/training';

const ACCEPTED = '.wav,.flac,.ogg,.mp3,.m4a,.aiff';
const POLL_ACTIVE_MS = 2000;
const POLL_IDLE_MS = 10000;

const TONE_CLASS = {
  ok: 'text-ok-fg',
  warn: 'text-warn-fg',
  danger: 'text-danger-fg',
  muted: 'text-fg-subtle',
};

const STATE_BADGE = {
  queued: 'border-line-strong text-fg-muted',
  running: 'border-ok-line text-ok-fg',
  done: 'border-ok-line-strong bg-ok-bg text-ok-fg',
  stopped: 'border-warn-line text-warn-fg',
  interrupted: 'border-warn-line text-warn-fg',
  error: 'border-danger-line text-danger-fg',
};

function Field({ label, htmlFor, hint, children }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-fg-muted">{label}</label>
      <div className="mt-2">{children}</div>
      {hint ? <p className="mt-1.5 text-[11px] leading-4 text-fg-faint">{hint}</p> : null}
    </div>
  );
}

const inputClass = 'w-full rounded-lg border border-line-strong bg-input px-3 py-2.5 text-sm text-fg '
  + 'focus:border-line-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 '
  + 'focus-visible:outline-[color:var(--frost-focus)] disabled:cursor-not-allowed disabled:opacity-50';

function JobCard({ job, onStop, onResume, onDelete, onUse, busy }) {
  const [moreEpochs, setMoreEpochs] = useState('');
  const active = isActiveJob(job);
  const progress = Math.round((job.progress || 0) * 100);
  const losses = job.losses || {};
  const canUse = Boolean(job.exported_epoch) && !active;
  return (
    <article className="rounded-lg border border-line-strong bg-raised px-4 py-4" aria-label={`Training ${job.name}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-fg">{job.name}</h3>
          <p className="mt-1 font-mono text-[11px] text-fg-faint">
            {formatDuration(job.audio_seconds)} audio · {job.sample_rate / 1000} kHz
            · {plannedEpochs(job) ? `${plannedEpochs(job)} epochs max` : 'automatic plan'}
            {job.segments ? ` · ${job.segments} segments` : ''}
          </p>
        </div>
        <span className={`rounded-sm border px-2 py-0.5 text-[11px] font-medium ${STATE_BADGE[job.state] || STATE_BADGE.queued}`}>
          {STATE_LABELS[job.state] || job.state}
        </span>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between gap-3 text-xs">
          <span className="text-fg-secondary">
            {active ? (STAGE_LABELS[job.stage] || job.message) : (job.message || STAGE_LABELS[job.stage])}
            {job.stage === 'train' && job.epoch ? ` · epoch ${job.epoch}/${plannedEpochs(job)}` : ''}
          </span>
          <span className="font-mono tabular-nums text-fg-muted">
            {progress}%{active && job.eta_seconds ? ` · ${formatDuration(job.eta_seconds)} left` : ''}
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-meter-track" role="progressbar"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
          <div className={`h-full transition-[width] duration-500 ${job.state === 'error' ? 'bg-danger-solid' : 'bg-ok-solid'}`}
            style={{ width: `${progress}%` }} />
        </div>
      </div>

      {job.stage === 'train' && Object.keys(losses).length ? (
        <dl className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums sm:grid-cols-6">
          {[['mel', 'Mel'], ['kl', 'KL'], ['fm', 'FM'], ['gen', 'Gen'], ['disc', 'Disc']].map(([key, label]) => (
            <div key={key} className="rounded-sm border border-line px-2 py-1">
              <dt className="text-fg-faint">{label}</dt>
              <dd className="text-fg-secondary">{losses[key]?.toFixed(2) ?? '–'}</dd>
            </div>
          ))}
          <div className="rounded-sm border border-line px-2 py-1">
            <dt className="text-fg-faint">Batch</dt>
            <dd className="text-fg-secondary">{job.batch_size ?? '–'} · {job.precision ?? ''}</dd>
          </div>
        </dl>
      ) : null}

      {job.warnings?.length ? (
        <ul className="mt-3 space-y-1 text-xs text-warn-fg">
          {job.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
      {job.error ? <p className="mt-3 break-words text-xs text-danger-fg">{job.error}</p> : null}
      {job.exported_epoch ? (
        <p className="mt-3 text-xs text-fg-subtle">
          {bestEpochSummary(job)} File: <span className="font-mono text-fg-secondary">{jobModelFile(job)}</span>
          {job.index ? ' with its index' : ''}
          {job.f0_stats?.median_hz ? ` · voice pitch ${Math.round(job.f0_stats.median_hz)} Hz (used by Auto pitch)` : ''}.
        </p>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {active ? (
          <button type="button" className="chip-button inline-flex items-center gap-1.5" onClick={() => onStop(job)} disabled={busy}>
            <Square className="h-3.5 w-3.5" aria-hidden="true" /> Stop and save
          </button>
        ) : null}
        {canUse ? (
          <button type="button" data-tone="primary" className="frost-control inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold"
            onClick={() => onUse(job)} disabled={busy}>
            <Wand2 className="h-3.5 w-3.5" aria-hidden="true" /> Use this voice
          </button>
        ) : null}
        {job.exported_epoch && !active ? (
          <a href={modelDownloadUrl(jobModelFile(job))} download className="chip-button inline-flex items-center gap-1.5"
            title="The trained voice is already saved in Models; this downloads a copy (.pth and .index)">
            <Download className="h-3.5 w-3.5" aria-hidden="true" /> Download voice (.zip)
          </a>
        ) : null}
        {canResume(job) ? (
          <>
            <input type="number" min={(job.epoch || 0) + 1} max={2000} placeholder={String(Math.max(plannedEpochs(job) || 0, (job.epoch || 0) + 100))}
              value={moreEpochs} onChange={(event) => setMoreEpochs(event.target.value)}
              className="w-24 rounded-md border border-line-strong bg-input px-2 py-1 text-xs text-fg"
              aria-label="Total epochs when continuing" />
            <button type="button" className="chip-button inline-flex items-center gap-1.5" disabled={busy}
              onClick={() => onResume(job, moreEpochs ? Number(moreEpochs) : null)}>
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" /> Continue training
            </button>
          </>
        ) : null}
        {!active ? (
          <button type="button" className="chip-button ml-auto inline-flex items-center gap-1.5" disabled={busy}
            onClick={() => onDelete(job)} aria-label={`Delete training ${job.name}`}>
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" /> Delete
          </button>
        ) : null}
      </div>
    </article>
  );
}

function TrainingStudio({ active = true, onActiveModelChange }) {
  const [info, setInfo] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [files, setFiles] = useState([]); // [{file, seconds}]
  const [name, setName] = useState('');
  const [sampleRate, setSampleRate] = useState(40000);
  const [epochs, setEpochs] = useState(300);
  // Automatic plan: epochs and batch from the audio and the GPU, early stopping, best epoch kept.
  const [autoPlan, setAutoPlan] = useState(true);
  const [batchSize, setBatchSize] = useState('');
  const [saveEvery, setSaveEvery] = useState(25);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const anyActive = jobs.some(isActiveJob);
  const totalSeconds = files.reduce((sum, item) => sum + (item.seconds || 0), 0);
  const quality = datasetQuality(totalSeconds, info);

  const refresh = useCallback(async () => {
    try {
      setJobs(await fetchTrainingJobs());
    } catch {
      // The server may be restarting; the next poll retries.
    }
  }, []);

  useEffect(() => {
    if (!active) return undefined;
    fetchTrainingInfo().then((value) => {
      setInfo(value);
      setEpochs(value.default_epochs);
      setSampleRate(value.default_sample_rate);
    }).catch(() => {});
    refresh();
    return undefined;
  }, [active, refresh]);

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(refresh, anyActive ? POLL_ACTIVE_MS : POLL_IDLE_MS);
    return () => clearInterval(timer);
  }, [active, anyActive, refresh]);

  const addFiles = async (list) => {
    const accepted = [...list].filter((file) => /\.(wav|flac|ogg|mp3|m4a|aiff?)$/i.test(file.name));
    if (accepted.length < list.length) setError('Some files were skipped: use WAV, FLAC, OGG, MP3 or M4A.');
    const measured = await Promise.all(accepted.map(async (file) => ({ file, seconds: await readAudioDuration(file) })));
    setFiles((previous) => [...previous, ...measured]);
    if (!name && accepted[0]) setName(accepted[0].name.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_'));
  };

  const start = async () => {
    setError(null);
    setNotice(null);
    setUploadProgress(0);
    try {
      await createTrainingJob({
        name, files: files.map((item) => item.file), sampleRate, epochs: autoPlan ? null : epochs,
        batchSize: !autoPlan && batchSize ? Number(batchSize) : null, saveEvery,
      }, setUploadProgress);
      setFiles([]);
      setNotice('Training started. You can keep using the studio; it will be slower while the GPU trains.');
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploadProgress(null);
    }
  };

  const act = async (action) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const useVoice = (job) => act(async () => {
    const result = await activateModel(jobModelFile(job));
    onActiveModelChange?.(result?.name || jobModelFile(job));
    setNotice(`${jobModelFile(job)} is now the active voice.`);
  });

  const canStart = files.length > 0 && name.trim() && uploadProgress === null && !anyActive
    && quality.tone !== 'danger';
  const fileSummary = useMemo(() => files.map(({ file, seconds }) => ({
    key: `${file.name}-${file.size}-${file.lastModified}`, name: file.name, size: file.size, seconds,
  })), [files]);

  return (
    <div className="space-y-5">
      <section className="panel p-5 sm:p-6" aria-labelledby="training-title">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="panel-kicker">Voice Lab training</p>
            <h2 id="training-title" className="panel-title">Create a voice</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-fg-muted">
              Upload recordings of one speaker. The studio cleans and slices them, learns the voice on your
              GPU starting from the official RVC v2 base models, builds its retrieval index, and saves a voice
              you can use live or in the converter.
            </p>
          </div>
          <GraduationCap className="hidden h-8 w-8 shrink-0 text-fg-subtle sm:block" strokeWidth={1.5} aria-hidden="true" />
        </div>

        {info?.device ? (
          <p className="mt-4 flex items-center gap-2 text-xs text-fg-secondary">
            <Cpu className="h-3.5 w-3.5 text-fg-subtle" aria-hidden="true" />
            Trains on {info.device.name}
            {info.device.device === 'cpu' ? <span className="text-warn-fg"> — no GPU found: training will be very slow</span> : null}
          </p>
        ) : null}

        <div
          role="button"
          tabIndex={0}
          aria-label="Add training audio"
          onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => { event.preventDefault(); setDragActive(false); addFiles(event.dataTransfer.files); }}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            fileInputRef.current?.click();
          }}
          className={`mt-5 cursor-pointer rounded-lg border border-dashed p-6 text-center transition-colors duration-150 ${
            dragActive ? 'border-line-hover bg-control-hover' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'
          }`}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPTED} multiple className="hidden"
            onChange={(event) => { addFiles(event.target.files); event.target.value = ''; }} />
          <Upload className="mx-auto mb-3 h-8 w-8 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />
          <p className="text-sm font-medium text-fg-secondary">Drop recordings or click to choose</p>
          <p className="mt-1 text-xs text-fg-faint">
            One speaker, dry and clean: no music, no reverb, no other voices. 5 minutes trains well; 10–30 minutes is ideal.
          </p>
        </div>

        {fileSummary.length ? (
          <ul className="mt-4 divide-y divide-line rounded-lg border border-line">
            {fileSummary.map((item, index) => (
              <li key={item.key} className="flex items-center justify-between gap-3 px-3 py-2 text-xs">
                <span className="min-w-0 truncate text-fg-secondary">{item.name}</span>
                <span className="flex shrink-0 items-center gap-3 font-mono text-fg-faint">
                  {item.seconds ? formatDuration(item.seconds) : '?'} · {formatBytes(item.size)}
                  <button type="button" className="text-fg-subtle hover:text-fg" aria-label={`Remove ${item.name}`}
                    onClick={() => setFiles((previous) => previous.filter((_, i) => i !== index))}>
                    <CircleX className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                </span>
              </li>
            ))}
          </ul>
        ) : null}
        <p className={`mt-3 text-xs ${TONE_CLASS[quality.tone]}`} role="status">
          {totalSeconds ? `${formatDuration(totalSeconds)} in total. ` : ''}{quality.text}
        </p>

        <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Voice name" htmlFor="training-name" hint="Saved as <name>.pth and <name>.index in Models.">
            <input id="training-name" className={inputClass} value={name} maxLength={60}
              onChange={(event) => setName(event.target.value)} placeholder="my_voice" />
          </Field>
          <Field label="Sample rate" htmlFor="training-rate" hint="40 kHz suits most microphones; 48 kHz for studio recordings.">
            <select id="training-rate" className={`native-select-safe ${inputClass}`} value={sampleRate}
              onChange={(event) => setSampleRate(Number(event.target.value))}>
              {(info?.sample_rates || [40000, 48000]).map((rate) => (
                <option key={rate} value={rate}>{rate / 1000} kHz{rate === 40000 ? ' (recommended)' : ''}</option>
              ))}
            </select>
          </Field>
          <Field label={autoPlan ? 'Epochs: automatic' : `Epochs: ${epochs}`} htmlFor="training-epochs"
            hint={autoPlan
              ? 'Planned from your audio and GPU. Training stops when the voice stops improving and keeps the best epoch.'
              : 'Upper limit; the best epoch is still the one kept. Stop any time: it saves.'}>
            <label className="mb-2 flex items-center gap-2 text-xs text-fg-secondary">
              <input type="checkbox" checked={autoPlan} onChange={(event) => setAutoPlan(event.target.checked)} />
              Automatic (recommended)
            </label>
            {!autoPlan ? (
              <input id="training-epochs" type="range" min={20} max={info?.max_epochs || 2000} step={10} value={epochs}
                onChange={(event) => setEpochs(Number(event.target.value))} className="fx-slider w-full" />
            ) : null}
          </Field>
          <Field label="Batch size" htmlFor="training-batch"
            hint={autoPlan ? 'Automatic: fits your GPU memory and the amount of audio.' : 'Empty = automatic from free GPU memory.'}>
            <input id="training-batch" type="number" min={1} max={32} className={inputClass} value={autoPlan ? '' : batchSize}
              disabled={autoPlan} onChange={(event) => setBatchSize(event.target.value)} placeholder="Auto" />
          </Field>
        </div>
        <details className="mt-4">
          <summary className="cursor-pointer text-xs font-medium text-fg-secondary">Advanced</summary>
          <div className="mt-3 max-w-xs">
            <Field label="Save every N epochs" htmlFor="training-save" hint="Checkpoint and update the usable voice this often.">
              <input id="training-save" type="number" min={1} max={500} className={inputClass} value={saveEvery}
                onChange={(event) => setSaveEvery(Number(event.target.value) || 25)} />
            </Field>
          </div>
        </details>

        {error ? <p className="mt-4 rounded-md border border-danger-line bg-danger-bg p-3 text-sm text-danger-fg" role="alert">{error}</p> : null}
        {notice ? <p className="mt-4 rounded-md border border-ok-line bg-ok-bg p-3 text-sm text-ok-fg" role="status">{notice}</p> : null}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button type="button" data-tone="primary" onClick={start} disabled={!canStart}
            className="frost-control inline-flex min-h-11 items-center gap-2 px-6 py-2.5 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-45">
            {uploadProgress !== null
              ? <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
              : <Play className="h-4 w-4" aria-hidden="true" />}
            {uploadProgress !== null ? `Uploading ${uploadProgress}%` : 'Start training'}
          </button>
          {anyActive ? <span className="text-xs text-fg-subtle">A voice is already training.</span> : null}
        </div>
      </section>

      <section className="panel p-5 sm:p-6" aria-labelledby="training-jobs-title">
        <p className="panel-kicker">Progress</p>
        <h2 id="training-jobs-title" className="panel-title">Trained voices</h2>
        <div className="mt-4 space-y-3">
          {jobs.length ? jobs.map((job) => (
            <JobCard key={job.id} job={job} busy={busy}
              onStop={(target) => act(() => stopTrainingJob(target.id))}
              onResume={(target, total) => act(() => resumeTrainingJob(target.id, total))}
              onDelete={(target) => act(() => deleteTrainingJob(target.id))}
              onUse={useVoice} />
          )) : <p className="py-6 text-center text-sm text-fg-subtle">No voices trained yet.</p>}
        </div>
      </section>
    </div>
  );
}

export default memo(TrainingStudio);
