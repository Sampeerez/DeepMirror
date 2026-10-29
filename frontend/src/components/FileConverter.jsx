import React, { memo, useEffect, useRef, useState } from 'react';
import { AudioLines, Download, FileAudio, FileVideo, LoaderCircle, X } from 'lucide-react';
import {
  convertFile, fetchModels, fetchVideoConversion, isAbortError, startVideoConversion, videoConversionUrl,
} from '../lib/api';
import { countActiveEffects } from '../lib/effects';
import { FALLBACK_F0_METHODS, findF0Method, groupedMethods } from '../lib/f0Methods';
import { formatBytes } from '../lib/format';
import Button from './Button';

const ACCEPTED = '.wav,.mp3,.flac,.ogg,.m4a,.aiff,.mp4,.mov,.mkv,.webm,.avi,.m4v';
const VIDEO_FILE = /\.(mp4|mov|mkv|webm|avi|m4v)$/i;
const VIDEO_STAGES = { queued: 'Starting', face: 'Swapping the face', voice: 'Converting the voice', mux: 'Joining' };

function isVideoFile(file) {
  return Boolean(file) && (file.type?.startsWith('video/') || VIDEO_FILE.test(file.name || ''));
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  const timer = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); });
});

const EFFECTS_ONLY = '';

// Offline conversion: an audio file through any voice in Models (the active one
// by default) + current studio settings, returned as a WAV; or a video file, with
// the face worn on every frame as well, returned as an MP4.
function FileConverter({ voice, effects, activeModel, f0Methods = FALLBACK_F0_METHODS, active = true, face = null }) {
  const [file, setFile] = useState(null);
  const [wearFace, setWearFace] = useState(true);
  const [progress, setProgress] = useState(null);   // {stage, fraction} while a video converts
  const [voices, setVoices] = useState([]);
  // null: follow the active voice; EFFECTS_ONLY or a model name once the user picks.
  const [chosenVoice, setChosenVoice] = useState(null);
  const [converting, setConverting] = useState(false);
  const [result, setResult] = useState(null); // {url, name, size, downloaded}
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [f0Override, setF0Override] = useState('studio');
  const fileInputRef = useRef(null);
  // Owns the in-flight request so Cancel can abort it.
  const abortRef = useRef(null);
  // Latest blob URL, for the unmount cleanup below.
  const resultUrlRef = useRef(null);
  useEffect(() => {
    resultUrlRef.current = result?.url || null;
  }, [result]);

  // The voice list is read whenever the tab is shown, so new and deleted voices appear.
  useEffect(() => {
    if (!active) return undefined;
    let cancelled = false;
    fetchModels()
      .then((list) => {
        if (cancelled) return;
        setVoices((Array.isArray(list) ? list : []).filter((item) => item?.type !== 'index').map((item) => item.name));
      })
      .catch(() => { if (!cancelled) setVoices([]); });
    return () => { cancelled = true; };
  }, [active]);

  const processWith = chosenVoice !== null && (chosenVoice === EFFECTS_ONLY || voices.includes(chosenVoice))
    ? chosenVoice
    : (activeModel || EFFECTS_ONLY);
  const fxCount = countActiveEffects(effects, voice.formant);
  const selectedF0 = findF0Method(f0Methods, f0Override === 'studio' ? voice.f0Method : f0Override);
  const offlineGroups = groupedMethods(f0Methods, 'offline');

  // Unmount: abandon a running request and free the last blob URL; neither
  // is reachable once the panel is gone.
  useEffect(() => () => {
    abortRef.current?.abort();
    if (resultUrlRef.current?.startsWith('blob:')) URL.revokeObjectURL(resultUrlRef.current);
  }, []);

  const pickFile = (candidate) => {
    if (!candidate || converting) return;
    setFile(candidate);
    setError(null);
    setNotice(null);
    setResult((prev) => {
      if (prev?.url?.startsWith('blob:')) URL.revokeObjectURL(prev.url);
      return null;
    });
  };

  const video = isVideoFile(file);
  const faceReady = Boolean(face?.ready && face?.activeFace);
  const useFace = video && wearFace && faceReady;

  const convertVideo = async (signal) => {
    let job = await startVideoConversion(file, {
      face: useFace, useModel: processWith !== EFFECTS_ONLY, model: processWith || null,
      pitchShift: voice.pitch, autoPitch: voice.autoPitch, formantShift: voice.formant, f0Method: selectedF0.id,
      effects, indexRate: voice.indexRate, protect: voice.protect, expressiveness: voice.expressiveness,
    });
    while (job.state !== 'done' && job.state !== 'error') {
      setProgress({ stage: VIDEO_STAGES[job.state] || 'Converting', fraction: job.progress });
      await sleep(1000, signal);
      job = await fetchVideoConversion(job.id);
    }
    if (job.state === 'error') throw new Error(job.error || 'Conversion failed');
    setResult({ url: videoConversionUrl(job.id), name: `${job.name}_converted.mp4`, size: null, video: true });
  };

  const handleConvert = async () => {
    if (!file || converting) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setConverting(true);
    setError(null);
    setNotice(null);
    try {
      if (video) {
        await convertVideo(controller.signal);
        return;
      }
      const blob = await convertFile(file, {
        pitchShift: voice.pitch,
        autoPitch: voice.autoPitch,
        expressiveness: voice.expressiveness,
        pitchCorrection: voice.pitchCorrection,
        formantShift: voice.formant,
        f0Method: selectedF0.id,
        effects,
        useModel: processWith !== EFFECTS_ONLY,
        model: processWith || null,
        // The panel promises "uses current studio settings", so the Voice Lab
        // advanced controls have to travel with the render too.
        indexRate: voice.indexRate,
        filterRadius: voice.filterRadius,
        rmsMixRate: voice.rmsMixRate,
        protect: voice.protect,
        crepeHopLength: voice.crepeHopLength,
        signal: controller.signal,
      });
      const name = `${file.name.replace(/\.[^.]+$/, '')}_converted.wav`;
      setResult((prev) => {
        if (prev?.url?.startsWith('blob:')) URL.revokeObjectURL(prev.url);
        return { url: URL.createObjectURL(blob), name, size: blob.size, downloaded: false };
      });
    } catch (err) {
      if (isAbortError(err)) {
        setNotice('Conversion cancelled.');
      } else {
        setError(err.message || 'Conversion failed');
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setConverting(false);
      setProgress(null);
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-5 lg:col-span-2" aria-labelledby="converter-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <p className="panel-kicker">Converter</p>
            <h2 id="converter-title" className="panel-title">Convert a recording or a video</h2>
          </div>
          <p className="text-xs text-fg-subtle">
            Studio settings: pitch {voice.pitch > 0 ? '+' : ''}{voice.pitch} st · formant {voice.formant > 0 ? '+' : ''}
            {voice.formant} st · {fxCount} effect{fxCount === 1 ? '' : 's'}
          </p>
        </div>

        {/* The file and how to convert it. */}
        <div className="mt-4 flex min-h-0 flex-1 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
          <div
            role="button"
            tabIndex={converting ? -1 : 0}
            aria-label="Choose an audio or video file to convert"
            aria-disabled={converting}
            onDrop={(event) => { event.preventDefault(); setDragActive(false); pickFile(event.dataTransfer.files?.[0]); }}
            onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }}
            onClick={() => { if (!converting) fileInputRef.current?.click(); }}
            onKeyDown={(event) => {
              if (converting || (event.key !== 'Enter' && event.key !== ' ')) return;
              event.preventDefault();
              fileInputRef.current?.click();
            }}
            className={`flex min-h-40 flex-col items-center justify-center rounded-lg border border-dashed p-6 text-center transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
              dragActive ? 'border-line-hover bg-control-hover drag-active' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'
            } ${converting ? 'pointer-events-none opacity-60' : 'cursor-pointer'}`}
          >
            <input ref={fileInputRef} type="file" accept={ACCEPTED} className="hidden"
              onChange={(event) => { pickFile(event.target.files?.[0]); if (fileInputRef.current) fileInputRef.current.value = ''; }} />
            {video
              ? <FileVideo className="mb-2 h-8 w-8 text-video-fg" strokeWidth={1.5} aria-hidden="true" />
              : <FileAudio className="mb-2 h-8 w-8 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />}
            {file ? (
              <>
                <p className="max-w-full truncate text-sm font-medium text-fg" title={file.name}>{file.name}</p>
                <p className="mt-1 text-xs text-fg-subtle">{formatBytes(file.size)} · click to choose another</p>
              </>
            ) : (
              <>
                <p className="text-sm font-medium text-fg-secondary">
                  {dragActive ? 'Drop here' : 'Drop or choose a file'}
                </p>
                <p className="mt-1 text-xs text-fg-faint">Audio: wav, mp3, flac, ogg, m4a · Video: mp4, mov, mkv, webm</p>
              </>
            )}
          </div>

          <div>
            <label htmlFor="converter-f0-method" className="block text-xs font-medium text-fg-muted">Pitch tracker</label>
            <select id="converter-f0-method" value={f0Override} disabled={converting}
              onChange={(event) => setF0Override(event.target.value)} className="native-select-safe field mt-2">
              <option value="studio">Studio setting: {findF0Method(f0Methods, voice.f0Method).label}</option>
              {offlineGroups.map((group) => (
                <optgroup key={group.section} label={group.section}>
                  {group.methods.map((method) => (
                    <option key={method.id} value={method.id} disabled={!method.available}>
                      {method.label}{method.available ? '' : ' (unavailable)'}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            <p className={`mt-1.5 text-[11px] leading-4 ${selectedF0.available ? 'text-fg-subtle' : 'text-warn-fg'}`}>
              {selectedF0.available ? selectedF0.description : selectedF0.reason}
            </p>
          </div>

          <div>
            <label htmlFor="converter-process" className="block text-xs font-medium text-fg-muted">Process with</label>
            <select id="converter-process" value={processWith} disabled={converting}
              onChange={(event) => setChosenVoice(event.target.value)} className="native-select-safe field mt-2">
              <option value={EFFECTS_ONLY}>Studio effects only</option>
              {voices.map((name) => (
                <option key={name} value={name}>{name}{name === activeModel ? ' (active)' : ''}</option>
              ))}
            </select>
            <p className="mt-1.5 text-[11px] leading-4 text-fg-subtle">
              {processWith && processWith !== activeModel
                ? 'Loaded only for this file; the live voice is not changed.'
                : 'Any voice in Models can convert the file.'}
            </p>
          </div>

          {video ? (
            <div>
              <label htmlFor="converter-face" className="block text-xs font-medium text-fg-muted">Face</label>
              <select id="converter-face" value={useFace ? 'on' : 'off'} disabled={converting || !faceReady}
                onChange={(event) => setWearFace(event.target.value === 'on')} className="native-select-safe field mt-2">
                {faceReady ? <option value="on">Wear {face.activeFace.name} on every frame</option> : null}
                <option value="off">Keep the faces as they are</option>
              </select>
              <p className="mt-1.5 text-[11px] leading-4 text-fg-subtle">
                {faceReady ? 'Every frame gets its own swap: the best quality, slower than live.'
                  : 'Choose a face in Video › Faces to swap it in the video.'}
              </p>
            </div>
          ) : null}

          <div className="mt-auto flex items-center gap-3">
            {converting ? (
              <Button variant="danger" size="lg" icon={X} onClick={handleCancel} className="w-40">Cancel</Button>
            ) : (
              <Button variant="primary" size="lg" icon={AudioLines} onClick={handleConvert} disabled={!file} className="w-40">
                Convert
              </Button>
            )}
            <p className="min-w-0 text-xs text-fg-subtle" role="status">
              {converting ? (
                <span className="flex items-center gap-2">
                  <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />
                  {progress ? `${progress.stage}… ${Math.round(progress.fraction * 100)}%` : 'Converting…'}
                </span>
              ) : notice || null}
            </p>
          </div>
        </div>
      </section>

      {/* The result. */}
      <section className="panel flex min-h-48 flex-col p-5" aria-labelledby="converter-result-title">
        <p className="panel-kicker">Converter</p>
        <h2 id="converter-result-title" className="panel-title">Result</h2>
        <div className="mt-4 flex min-h-0 flex-1 flex-col rounded-lg border border-line bg-raised p-4">
          {error ? (
            <p className="break-words rounded-md border border-danger-line bg-danger-bg p-3 text-sm text-danger-fg" role="alert">
              {error}
            </p>
          ) : null}
          {result ? (
            <>
              <p className="text-xs font-medium text-fg-muted">Result</p>
              <p className="mt-1 truncate text-sm font-semibold text-fg" title={result.name}>{result.name}</p>
              {result.size ? <p className="text-xs text-fg-subtle">{formatBytes(result.size)}</p> : null}
              {result.video ? (
                <video controls src={result.url} className="mt-4 aspect-video w-full rounded-md bg-black"
                  aria-label={`Play ${result.name}`} />
              ) : (
                <audio controls src={result.url} className="mt-4 h-10 w-full" aria-label={`Play ${result.name}`} />
              )}
              <div className="mt-4">
                <Button href={result.url} download={result.name} icon={Download} size="md"
                  onClick={() => setResult((current) => current?.url === result.url ? { ...current, downloaded: true } : current)}>
                  {result.video ? 'Download MP4' : 'Download WAV'}
                </Button>
              </div>
            </>
          ) : !error ? (
            <div className="m-auto flex flex-col items-center gap-2 text-center text-sm text-fg-subtle">
              <AudioLines className="h-8 w-8 text-fg-faint" strokeWidth={1.5} aria-hidden="true" />
              The result appears here.
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export default memo(FileConverter);
