import React, { memo, useEffect, useRef, useState } from 'react';
import { AudioLines, Boxes, Download, FileAudio, FileVideo, LoaderCircle, UploadCloud, X } from 'lucide-react';
import {
  convertFile, faceThumbnailUrl, fetchModels, fetchVideoConversion, isAbortError, startVideoConversion,
  videoConversionUrl,
} from '../lib/api';
import { FALLBACK_F0_METHODS, findF0Method } from '../lib/f0Methods';
import { formatBytes } from '../lib/format';
import Button from './Button';
import Picker from './Picker';
import Toggle from './Toggle';

const ACCEPT = '.wav,.mp3,.flac,.ogg,.m4a,.aiff,.mp4,.mov,.mkv,.webm,.avi,.m4v';
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

// Library › Convert: drop an audio file (back as a WAV in another voice) or a
// video (back as an MP4, with the face worn on every frame too). Any voice in
// Voices can convert it; the rest of Live's settings come along.
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
  const selectedF0 = findF0Method(f0Methods, voice.f0Method);

  // Unmount: abandon a running request and free the last blob URL; neither
  // is reachable once the panel is gone.
  useEffect(() => () => {
    abortRef.current?.abort();
    if (resultUrlRef.current?.startsWith('blob:')) URL.revokeObjectURL(resultUrlRef.current);
  }, []);

  const pickFile = (candidate) => {
    if (converting) return;
    if (!candidate) {
      setFile(null);
      return;
    }
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

  const busyText = converting
    ? (progress ? `${progress.stage}… ${Math.round(progress.fraction * 100)}%` : 'Converting…') : notice;

  return (
    <section className="panel flex min-h-0 flex-col p-4" aria-labelledby="converter-title">
      <h2 id="converter-title" className="flex h-7 shrink-0 items-center text-sm font-semibold text-fg">Convert a file</h2>

      <div className="mt-3 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        <div
          role="button"
          tabIndex={converting ? -1 : 0}
          aria-label="Choose an audio or video file"
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
          className={`flex min-h-36 shrink-0 flex-col items-center justify-center rounded-lg border border-dashed p-5 text-center transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
            dragActive ? 'border-brand-line bg-brand-bg drag-active' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'
          } ${converting ? 'pointer-events-none opacity-60' : 'cursor-pointer'}`}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPT} className="hidden"
            onChange={(event) => { pickFile(event.target.files?.[0]); if (fileInputRef.current) fileInputRef.current.value = ''; }} />
          {file ? (
            <>
              {video
                ? <FileVideo className="mb-2 h-7 w-7 text-video-fg" strokeWidth={1.5} aria-hidden="true" />
                : <FileAudio className="mb-2 h-7 w-7 text-ok-fg" strokeWidth={1.5} aria-hidden="true" />}
              <p className="max-w-full truncate text-sm font-medium text-fg" title={file.name}>{file.name}</p>
              <p className="mt-1 text-xs text-fg-subtle">{video ? 'Video' : 'Audio'} · {formatBytes(file.size)}</p>
            </>
          ) : (
            <>
              <UploadCloud className="mb-2 h-7 w-7 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />
              <p className="text-sm font-medium text-fg-secondary">{dragActive ? 'Drop here' : 'Drop an audio or video file'}</p>
              <p className="mt-1 text-xs text-fg-faint">wav · mp3 · flac · m4a · mp4 · mov · mkv · webm</p>
            </>
          )}
        </div>

        <Picker label="Voice" icon={Boxes} accent="voice" value={processWith} disabled={converting}
          onChange={setChosenVoice}
          options={[{ value: EFFECTS_ONLY, label: 'No voice (effects only)' },
            ...voices.map((name) => ({ value: name, label: name.replace(/\.(pth|pt)$/i, ''),
              note: name === activeModel ? 'live' : null }))]} />

        {video ? (
          <div className="flex h-12 shrink-0 items-center gap-3 rounded-lg border border-line bg-raised px-3">
            {faceReady ? (
              <img src={faceThumbnailUrl(face.activeFace.id)} alt="" className="h-8 w-8 shrink-0 rounded-md object-cover" />
            ) : null}
            <span className="min-w-0 flex-1 truncate text-sm text-fg-secondary">
              {faceReady ? `Wear ${face.activeFace.name}` : 'No face chosen in Faces'}
            </span>
            <Toggle checked={useFace} disabled={converting || !faceReady} onChange={setWearFace} label="Wear the face" />
          </div>
        ) : null}

        <div className="flex shrink-0 items-center gap-3">
          {converting ? (
            <Button variant="danger" size="lg" icon={X} onClick={handleCancel} className="w-36">Cancel</Button>
          ) : (
            <Button variant="primary" size="lg" icon={AudioLines} onClick={handleConvert} disabled={!file} className="w-36">
              Convert
            </Button>
          )}
          <p className="flex min-w-0 items-center gap-2 text-xs text-fg-subtle" role="status">
            {converting ? <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" /> : null}
            {busyText}
          </p>
        </div>

        {error ? (
          <p className="break-words rounded-md border border-danger-line bg-danger-bg p-3 text-sm text-danger-fg" role="alert">
            {error}
          </p>
        ) : null}
        {result ? (
          <div className="shrink-0 rounded-lg border border-line bg-raised p-3">
            <div className="flex items-center justify-between gap-3">
              <p className="min-w-0 truncate text-sm font-medium text-fg" title={result.name}>{result.name}</p>
              <Button href={result.url} download={result.name} icon={Download}
                onClick={() => setResult((current) => current?.url === result.url ? { ...current, downloaded: true } : current)}>
                {result.video ? 'MP4' : 'WAV'}
              </Button>
            </div>
            {result.video ? (
              <video controls src={result.url} className="mt-3 aspect-video w-full rounded-md bg-black"
                aria-label={`Play ${result.name}`} />
            ) : (
              <audio controls src={result.url} className="mt-3 h-10 w-full" aria-label={`Play ${result.name}`} />
            )}
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default memo(FileConverter);
