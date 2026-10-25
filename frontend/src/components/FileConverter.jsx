import React, { memo, useEffect, useRef, useState } from 'react';
import { AudioLines, Download, FileAudio, LoaderCircle, X } from 'lucide-react';
import { convertFile, isAbortError } from '../lib/api';
import { countActiveEffects } from '../lib/effects';
import { FALLBACK_F0_METHODS, findF0Method, groupedMethods } from '../lib/f0Methods';
import { formatBytes } from '../lib/format';
import Button from './Button';

const ACCEPTED = '.wav,.mp3,.flac,.ogg,.m4a,.aiff';

// Offline conversion: renders an uploaded audio file through the active
// model + current studio settings on the server, returns a WAV.
function FileConverter({ voice, effects, activeModel, f0Methods = FALLBACK_F0_METHODS }) {
  const [file, setFile] = useState(null);
  const [useModel, setUseModel] = useState(true);
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

  const fxCount = countActiveEffects(effects, voice.formant);
  const selectedF0 = findF0Method(f0Methods, f0Override === 'studio' ? voice.f0Method : f0Override);
  const offlineGroups = groupedMethods(f0Methods, 'offline');

  // Unmount: abandon a running request and free the last blob URL; neither
  // is reachable once the panel is gone.
  useEffect(() => () => {
    abortRef.current?.abort();
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current);
  }, []);

  const pickFile = (candidate) => {
    if (!candidate || converting) return;
    setFile(candidate);
    setError(null);
    setNotice(null);
    setResult((prev) => {
      if (prev?.url) URL.revokeObjectURL(prev.url);
      return null;
    });
  };

  const handleConvert = async () => {
    if (!file || converting) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setConverting(true);
    setError(null);
    setNotice(null);
    try {
      const blob = await convertFile(file, {
        pitchShift: voice.pitch,
        autoPitch: voice.autoPitch,
        expressiveness: voice.expressiveness,
        pitchCorrection: voice.pitchCorrection,
        formantShift: voice.formant,
        f0Method: selectedF0.id,
        effects,
        useModel: useModel && Boolean(activeModel),
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
        if (prev?.url) URL.revokeObjectURL(prev.url);
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
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  return (
    <section className="panel mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col p-5" aria-labelledby="converter-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <p className="panel-kicker">Converter</p>
          <h2 id="converter-title" className="panel-title">Convert a recording</h2>
        </div>
        <p className="text-xs text-fg-subtle">
          Studio settings: pitch {voice.pitch > 0 ? '+' : ''}{voice.pitch} st · formant {voice.formant > 0 ? '+' : ''}
          {voice.formant} st · {fxCount} effect{fxCount === 1 ? '' : 's'}
        </p>
      </div>

      <div className="mt-4 grid min-h-0 flex-1 gap-5 lg:grid-cols-2">
        {/* Left: the file and how to convert it. */}
        <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
          <div
            role="button"
            tabIndex={converting ? -1 : 0}
            aria-label="Choose an audio file to convert"
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
            <FileAudio className="mb-2 h-8 w-8 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />
            {file ? (
              <>
                <p className="max-w-full truncate text-sm font-medium text-fg" title={file.name}>{file.name}</p>
                <p className="mt-1 text-xs text-fg-subtle">{formatBytes(file.size)} · click to choose another</p>
              </>
            ) : (
              <>
                <p className="text-sm font-medium text-fg-secondary">
                  {dragActive ? 'Drop the audio file here' : 'Drop an audio file or click to choose'}
                </p>
                <p className="mt-1 text-xs text-fg-faint">wav · mp3 · flac · ogg · m4a</p>
              </>
            )}
          </div>

          <div>
            <label htmlFor="converter-f0-method" className="block text-xs font-medium text-fg-muted">Pitch tracker</label>
            <select id="converter-f0-method" value={f0Override} disabled={converting}
              onChange={(event) => setF0Override(event.target.value)} className="native-select-safe field mt-2">
              <option value="studio">Studio setting — {findF0Method(f0Methods, voice.f0Method).label}</option>
              {offlineGroups.map((group) => (
                <optgroup key={group.section} label={group.section}>
                  {group.methods.map((method) => (
                    <option key={method.id} value={method.id} disabled={!method.available}>
                      {method.label}{method.available ? '' : ' — unavailable'}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
            <p className={`mt-1.5 text-[11px] leading-4 ${selectedF0.available ? 'text-fg-subtle' : 'text-warn-fg'}`}>
              {selectedF0.available ? selectedF0.description : selectedF0.reason}
            </p>
          </div>

          <label className={`flex min-w-0 items-center gap-2.5 text-sm ${activeModel ? 'text-fg-secondary' : 'text-fg-faint'}`}>
            <input type="checkbox" checked={useModel && Boolean(activeModel)} disabled={!activeModel || converting}
              onChange={(event) => setUseModel(event.target.checked)} className="h-4 w-4 shrink-0 accent-primary" />
            <span className="shrink-0">Use the active voice</span>
            <span className="min-w-0 truncate text-xs text-fg-faint" title={activeModel || undefined}>
              {activeModel ? `(${activeModel})` : '(none active — effects only)'}
            </span>
          </label>

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
                  Converting… long files take a while.
                </span>
              ) : notice || null}
            </p>
          </div>
        </div>

        {/* Right: the result. */}
        <div className="flex min-h-48 flex-col rounded-lg border border-line bg-raised p-4">
          {error ? (
            <p className="break-words rounded-md border border-danger-line bg-danger-bg p-3 text-sm text-danger-fg" role="alert">
              {error}
            </p>
          ) : null}
          {result ? (
            <>
              <p className="text-xs font-medium text-fg-muted">Result</p>
              <p className="mt-1 truncate text-sm font-semibold text-fg" title={result.name}>{result.name}</p>
              <p className="text-xs text-fg-subtle">{formatBytes(result.size)}</p>
              <audio controls src={result.url} className="mt-4 h-10 w-full" aria-label={`Play ${result.name}`} />
              <div className="mt-4">
                <Button href={result.url} download={result.name} icon={Download} size="md"
                  onClick={() => setResult((current) => current?.url === result.url ? { ...current, downloaded: true } : current)}>
                  Download WAV
                </Button>
              </div>
            </>
          ) : !error ? (
            <div className="m-auto flex flex-col items-center gap-2 text-center text-sm text-fg-subtle">
              <AudioLines className="h-8 w-8 text-fg-faint" strokeWidth={1.5} aria-hidden="true" />
              The converted recording appears here, ready to play and download.
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default memo(FileConverter);
