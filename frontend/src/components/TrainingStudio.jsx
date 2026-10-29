import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CircleX, Cpu, Download, GraduationCap, Play, RotateCcw, Square, Trash2, Upload, Wand2,
} from 'lucide-react';
import {
  activateModel, createTrainingJob, deleteTrainingJob, fetchTrainingInfo, fetchTrainingJobs, modelDownloadUrl,
  resumeTrainingJob, stopTrainingJob,
} from '../lib/api';
import { formatBytes } from '../lib/format';
import Button from './Button';
import Segmented from './Segmented';
import {
  STAGE_LABELS, STATE_LABELS, bestEpochSummary, canResume, datasetQuality, formatDuration, isActiveJob, jobModelFile,
  plannedEpochs, readAudioDuration,
} from '../lib/training';

const ACCEPTED = '.wav,.flac,.ogg,.mp3,.m4a,.aiff';
const BASE_LABELS = { official: 'RVC v2 (official)', titan: 'TITAN Medium' };
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
      <div className="mt-1.5">{children}</div>
      {hint ? <p className="mt-1 truncate text-[11px] leading-4 text-fg-faint" title={hint}>{hint}</p> : null}
    </div>
  );
}

const inputClass = 'field';

function JobCard({ job, onStop, onResume, onDelete, onUse, busy }) {
  const [moreEpochs, setMoreEpochs] = useState('');
  const active = isActiveJob(job);
  const progress = Math.round((job.progress || 0) * 100);
  const losses = job.losses || {};
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
        <span className={`shrink-0 whitespace-nowrap rounded-sm border px-2 py-0.5 text-[11px] font-medium ${STATE_BADGE[job.state] || STATE_BADGE.queued}`}>
          {STATE_LABELS[job.state] || job.state}
        </span>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between gap-3 text-xs">
          <span className="min-w-0 truncate text-fg-secondary" title={job.message || undefined}>
            {active ? (STAGE_LABELS[job.stage] || job.message) : (job.message || STAGE_LABELS[job.stage])}
            {job.stage === 'train' && job.epoch ? ` · epoch ${job.epoch}/${plannedEpochs(job)}` : ''}
          </span>
          <span className="shrink-0 font-mono tabular-nums text-fg-muted">
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
        <dl className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
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
          {job.warnings.map((warning) => <li key={warning} className="break-words">{warning}</li>)}
        </ul>
      ) : null}
      {job.error ? <p className="mt-3 break-words text-xs text-danger-fg">{job.error}</p> : null}
      {job.exported_epoch ? (
        <p className="mt-3 break-words text-xs text-fg-subtle">
          {bestEpochSummary(job)}
          {job.f0_stats?.median_hz ? ` · ${Math.round(job.f0_stats.median_hz)} Hz` : ''}
        </p>
      ) : null}

      <div className="mt-4 space-y-2">
        {active ? (
          <Button size="md" icon={Square} onClick={() => onStop(job)} disabled={busy} className="w-full">Stop and save</Button>
        ) : null}
        {job.exported_epoch && !active ? (
          <div className="grid grid-cols-2 gap-2">
            <Button size="md" variant="primary" icon={Wand2} onClick={() => onUse(job)} disabled={busy}>
              Use this voice
            </Button>
            <Button size="md" href={modelDownloadUrl(jobModelFile(job))} download icon={Download}>
              Download
            </Button>
          </div>
        ) : null}
        {!active ? (
          <div className="flex items-center gap-2">
            {canResume(job) ? (
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <input type="number" min={(job.epoch || 0) + 1} max={2000}
                  placeholder={String(Math.max(plannedEpochs(job) || 0, (job.epoch || 0) + 100))}
                  value={moreEpochs} onChange={(event) => setMoreEpochs(event.target.value)}
                  className="field w-24 shrink-0" aria-label="Total epochs when continuing" title="Total epochs" />
                <Button size="md" icon={RotateCcw} disabled={busy} className="min-w-0 flex-1"
                  onClick={() => onResume(job, moreEpochs ? Number(moreEpochs) : null)}>
                  Continue
                </Button>
              </div>
            ) : <div className="flex-1" />}
            <Button size="md" variant="danger" icon={Trash2} disabled={busy} onClick={() => onDelete(job)}
              aria-label={`Delete training ${job.name}`} title="Delete this training" />
          </div>
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
  const [base, setBase] = useState(null);
  const [trainingQuality, setTrainingQuality] = useState('high');
  const [isolate, setIsolate] = useState(true);
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
      setBase((current) => current || value.default_base || 'official');
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
        batchSize: !autoPlan && batchSize ? Number(batchSize) : null, saveEvery, base, quality: trainingQuality, isolate,
      }, setUploadProgress);
      setFiles([]);
      setNotice('Training started. The studio is slower while the GPU trains.');
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
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-4 lg:col-span-2" aria-labelledby="training-title">
        <h2 id="training-title" className="panel-title">Create a voice</h2>
        <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">

        {info?.device ? (
          <p className="flex items-center gap-2 text-xs text-fg-secondary">
            <Cpu className="h-3.5 w-3.5 text-fg-subtle" aria-hidden="true" />
            Trains on {info.device.name}
            {info.device.device === 'cpu' ? <span className="text-warn-fg">: no GPU found: training will be very slow</span> : null}
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
          className={`mt-3 flex cursor-pointer items-center justify-center gap-3 rounded-lg border border-dashed px-5 py-3.5 transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
            dragActive ? 'border-line-hover bg-control-hover' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'
          }`}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPTED} multiple className="hidden"
            onChange={(event) => { addFiles(event.target.files); event.target.value = ''; }} />
          <Upload className="h-6 w-6 shrink-0 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />
          <div className="min-w-0 text-left">
            <p className="text-sm font-medium text-fg-secondary">Drop recordings or click to choose</p>
            <p className="mt-0.5 text-xs text-fg-faint">One speaker · 10–30 min is best</p>
          </div>
        </div>

        {fileSummary.length ? (
          <ul className="mt-4 divide-y divide-line rounded-lg border border-line">
            {fileSummary.map((item, index) => (
              <li key={item.key} className="flex items-center justify-between gap-3 px-3 py-2 text-xs">
                <span className="min-w-0 truncate text-fg-secondary" title={item.name}>{item.name}</span>
                <span className="flex shrink-0 items-center gap-2 font-mono text-fg-faint">
                  {item.seconds ? formatDuration(item.seconds) : '?'} · {formatBytes(item.size)}
                  <Button icon={CircleX} variant="ghost" aria-label={`Remove ${item.name}`} title="Remove"
                    onClick={() => setFiles((previous) => previous.filter((_, i) => i !== index))} />
                </span>
              </li>
            ))}
          </ul>
        ) : null}
        {totalSeconds ? (
          <p className={`mt-2 text-xs ${TONE_CLASS[quality.tone]}`} role="status">
            {formatDuration(totalSeconds)} · {quality.text}
          </p>
        ) : null}

        <div className="mt-3 grid gap-x-4 gap-y-3 sm:grid-cols-2">
          <Field label="Voice name" htmlFor="training-name">
            <input id="training-name" className={inputClass} value={name} maxLength={60}
              onChange={(event) => setName(event.target.value)} placeholder="my_voice" />
          </Field>
          <Field label="Sample rate">
            <Segmented label="Sample rate" hideLabel accent="voice" size="sm" value={sampleRate} onChange={setSampleRate}
              options={(info?.sample_rates || [40000, 48000]).map((rate) => ({ value: rate, label: `${rate / 1000} kHz` }))} />
          </Field>
          <Field label={autoPlan ? 'Epochs' : `Epochs: ${epochs}`}>
            <Segmented label="Epochs" hideLabel accent="voice" size="sm" value={autoPlan ? 'auto' : 'manual'}
              onChange={(value) => setAutoPlan(value === 'auto')}
              options={[{ value: 'auto', label: 'Automatic' }, { value: 'manual', label: 'Manual' }]} />
            {!autoPlan ? (
              <input id="training-epochs" type="range" min={20} max={info?.max_epochs || 2000} step={10} value={epochs}
                aria-label="Epochs" onChange={(event) => setEpochs(Number(event.target.value))}
                className="fx-slider mt-3 w-full" />
            ) : null}
          </Field>
          <Field label="Batch size" htmlFor="training-batch">
            <input id="training-batch" type="number" min={1} max={32} className={inputClass} value={autoPlan ? '' : batchSize}
              disabled={autoPlan} onChange={(event) => setBatchSize(event.target.value)} placeholder="Auto" />
          </Field>
          <Field label="Training quality">
            <Segmented label="Training quality" hideLabel accent="voice" size="sm" value={trainingQuality}
              onChange={setTrainingQuality}
              options={[{ value: 'high', label: 'High' }, { value: 'standard', label: 'Standard' }]} />
          </Field>
          <Field label="Background">
            <Segmented label="Background" hideLabel accent="voice" size="sm" value={isolate ? 'voice' : 'as-is'}
              onChange={(value) => setIsolate(value === 'voice')}
              options={[{ value: 'voice', label: 'Voice only' }, { value: 'as-is', label: 'As it is' }]} />
          </Field>
          <Field label="Base model">
            <Segmented label="Base model" hideLabel accent="voice" size="sm" value={base || 'official'} onChange={setBase}
              options={(info?.bases || ['official', 'titan']).map((id) => ({ value: id, label: BASE_LABELS[id] || id }))} />
          </Field>
          <Field label="Save every N epochs" htmlFor="training-save">
            <input id="training-save" type="number" min={1} max={500} className={inputClass} value={saveEvery}
              onChange={(event) => setSaveEvery(Number(event.target.value) || 25)} />
          </Field>
        </div>

        {error ? <p className="mt-4 break-words rounded-md border border-danger-line bg-danger-bg p-3 text-sm text-danger-fg" role="alert">{error}</p> : null}
        {notice ? <p className="mt-4 rounded-md border border-ok-line bg-ok-bg p-3 text-sm text-ok-fg" role="status">{notice}</p> : null}

        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-line pt-3">
          <Button variant="primary" size="lg" icon={Play} onClick={start} disabled={!canStart}
            loading={uploadProgress !== null} className="w-48">
            {uploadProgress !== null ? `Uploading ${uploadProgress}%` : 'Start training'}
          </Button>
          {anyActive ? <span className="text-xs text-fg-subtle">A voice is already training.</span> : null}
        </div>
      </section>

      <section className="panel flex min-h-0 flex-col p-4" aria-labelledby="training-jobs-title">
        <h2 id="training-jobs-title" className="panel-title">Trained voices</h2>
        <div className="mt-4 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
          {jobs.length ? jobs.map((job) => (
            <JobCard key={job.id} job={job} busy={busy}
              onStop={(target) => act(() => stopTrainingJob(target.id))}
              onResume={(target, total) => act(() => resumeTrainingJob(target.id, total))}
              onDelete={(target) => act(() => deleteTrainingJob(target.id))}
              onUse={useVoice} />
          )) : (
            <div className="flex flex-col items-center gap-2 py-12 text-center text-sm text-fg-subtle">
              <GraduationCap className="h-8 w-8 text-fg-faint" strokeWidth={1.5} aria-hidden="true" />
              No voices trained yet.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default memo(TrainingStudio);
