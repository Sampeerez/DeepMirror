import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Boxes, CircleX, Download, LoaderCircle, RefreshCw, Search, Trash2, Upload } from 'lucide-react';
import {
  modelDownloadUrl,
  fetchModels,
  uploadModel,
  deleteModel,
  activateModel,
  deactivateModel,
  getActiveModel,
} from '../lib/api';
import { formatBytes } from '../lib/format';
import Button from './Button';

// A "Confirm" that stays armed forever is a trap for the next stray click.
const DELETE_CONFIRM_MS = 4000;

function getModelType(name) {
  if (name.endsWith('.pth') || name.endsWith('.pt')) return 'RVC';
  return 'Unknown';
}

function MetaBadge({ children, tone = 'default' }) {
  const tones = {
    default: 'border-line-strong bg-control text-fg-subtle',
    emerald: 'border-ok-line bg-ok-bg text-ok-fg',
  };
  return (
    <span className={`whitespace-nowrap rounded-sm border px-2 py-0.5 text-[11px] font-medium ${tones[tone] || tones.default}`}>
      {children}
    </span>
  );
}

// The active model is owned by the parent: this panel reports what the server
// says through `onActiveModelChange` and renders `activeModel` as given, so
// there is one source of truth instead of a local mirror to keep in step.
//
// `active` is true while the Models tab is showing. The panel stays mounted
// across tab switches, so this is what triggers a fresh listing on each visit.
function ModelManager({ activeModel = null, onActiveModelChange, active = true, loadingModel = null, engine = null }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [operatingOn, setOperatingOn] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [search, setSearch] = useState('');

  const fileInputRef = useRef(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const [modelList, active] = await Promise.all([
        fetchModels(),
        getActiveModel(),
      ]);
      setModels(Array.isArray(modelList) ? modelList : []);
      onActiveModelChange?.(active?.name || active?.model || null);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  }, [onActiveModelChange]);

  useEffect(() => {
    if (active) loadModels();
  }, [active, loadModels]);

  // Disarm a pending delete confirmation after a moment.
  useEffect(() => {
    if (!deleteConfirm) return undefined;
    const timer = setTimeout(() => setDeleteConfirm(null), DELETE_CONFIRM_MS);
    return () => clearTimeout(timer);
  }, [deleteConfirm]);

  const syncActiveModel = useCallback((name) => {
    onActiveModelChange?.(name);
  }, [onActiveModelChange]);

  const validateUploadFile = (file) => {
    const validExtensions = ['.pth', '.pt', '.index'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
      return 'Invalid file type. Upload an RVC .pth model and, optionally, its .index file.';
    }
    return null;
  };

  // Uploaded one at a time: a single progress bar can only honestly track one
  // transfer, and checkpoints are large enough that parallel uploads just
  // compete for the same bandwidth.
  const handleUploadAll = async (files) => {
    if (uploading || files.length === 0) return;

    const validationError = files.map(validateUploadFile).find(Boolean);
    if (validationError) {
      setError(validationError);
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      for (const file of files) {
        setUploadProgress(0);
        // Real byte-level transport progress from the XHR upload stream.
        await uploadModel(file, setUploadProgress);
        setUploadProgress(100);
        // Keep the model list truthful even if a later file fails.
        await loadModels();
      }
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    handleUploadAll(files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files || []);
    handleUploadAll(files);
  };

  const handleActivate = async (name) => {
    setOperatingOn(name);
    setError(null);
    try {
      await activateModel(name);
      syncActiveModel(name);
      await loadModels();
    } catch (err) {
      setError(err.message || 'Failed to activate model');
    } finally {
      setOperatingOn(null);
    }
  };

  const handleDeactivate = async () => {
    setOperatingOn(activeModel);
    setError(null);
    try {
      await deactivateModel();
      syncActiveModel(null);
    } catch (err) {
      setError(err.message || 'Failed to deactivate model');
    } finally {
      setOperatingOn(null);
    }
  };

  const handleDelete = async (name) => {
    if (deleteConfirm !== name) {
      setDeleteConfirm(name);
      return;
    }

    setDeleteConfirm(null);
    setOperatingOn(name);
    setError(null);

    try {
      if (activeModel === name) {
        await deactivateModel();
        syncActiveModel(null);
      }
      await deleteModel(name);
      await loadModels();
    } catch (err) {
      setError(err.message || 'Failed to delete model');
    } finally {
      setOperatingOn(null);
    }
  };

  const query = search.trim().toLowerCase();
  const rows = models
    .map((model) => (typeof model === 'string' ? { name: model } : model))
    .filter((model) => !query || model.name.toLowerCase().includes(query))
    // The active voice first, then alphabetical.
    .sort((a, b) => (b.name === activeModel) - (a.name === activeModel) || a.name.localeCompare(b.name));

  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto">
        <section className="panel p-5" aria-labelledby="models-add-title">
          <p className="panel-kicker">Models</p>
          <h2 id="models-add-title" className="panel-title">Add a voice</h2>
          <div
            role="button"
            tabIndex={uploading ? -1 : 0}
            aria-label="Upload voice model files"
            aria-disabled={uploading}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (uploading || (event.key !== 'Enter' && event.key !== ' ')) return;
              event.preventDefault();
              fileInputRef.current?.click();
            }}
            className={`relative mt-4 cursor-pointer rounded-lg border border-dashed p-5 text-center transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
              dragActive ? 'border-line-hover bg-control-hover drag-active' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'
            } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
          >
            <input ref={fileInputRef} type="file" accept=".pth,.pt,.index" multiple onChange={handleFileSelect}
              className="hidden" />
            <Upload className="mx-auto mb-2 h-7 w-7 text-fg-subtle" strokeWidth={1.5} aria-hidden="true" />
            <p className="text-sm font-medium text-fg-secondary">
              {dragActive ? 'Drop here' : 'Drop or choose files'}
            </p>
            <p className="mt-1 text-[11px] text-fg-faint">.pth / .pt + optional .index</p>
            {uploading ? (
              <>
                <p className="mt-2 font-mono text-[11px] tabular-nums text-fg-muted" role="status">
                  {uploadProgress >= 99 ? 'Finalizing on the server…' : `Uploading… ${uploadProgress}%`}
                </p>
                <div className="absolute inset-x-0 bottom-0 h-1 overflow-hidden rounded-b-lg bg-meter-track">
                  <div className="h-full bg-primary" style={{ width: `${uploadProgress}%` }} />
                </div>
              </>
            ) : null}
          </div>
          {error ? (
            <div className="mt-3 flex items-start gap-2 rounded-md border border-danger-line bg-danger-bg p-3" role="alert">
              <CircleX className="mt-0.5 h-4 w-4 shrink-0 text-danger-fg" aria-hidden="true" />
              <p className="min-w-0 break-words text-xs text-danger-fg">{error}</p>
            </div>
          ) : null}
        </section>
        {engine ? <div className="panel px-5">{engine}</div> : null}
      </div>

      <section className="panel flex min-h-0 flex-col p-5" aria-labelledby="models-list-title">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="panel-kicker">Library</p>
            <h2 id="models-list-title" className="panel-title">
              Voices <span className="font-normal text-fg-subtle">· {models.length}</span>
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <label className="relative">
              <span className="sr-only">Search voices</span>
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-faint"
                aria-hidden="true" />
              <input type="search" value={search} onChange={(event) => setSearch(event.target.value)}
                placeholder="Search" className="field w-44 pl-8 sm:w-56" />
            </label>
            <Button icon={RefreshCw} size="md" onClick={loadModels} disabled={loading} aria-label="Refresh models"
              title="Refresh models" className={loading ? '[&_svg]:animate-spin' : ''} />
          </div>
        </div>

        <ul className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1" aria-label="Voice models">
          {loading && models.length === 0 ? (
            <li className="flex items-center justify-center py-10">
              <LoaderCircle className="h-5 w-5 animate-spin text-fg-subtle" aria-label="Loading" />
            </li>
          ) : rows.length === 0 ? (
            <li className="flex flex-col items-center gap-2 px-4 py-12 text-center text-sm text-fg-subtle">
              <Boxes className="h-8 w-8 text-fg-faint" strokeWidth={1.5} aria-hidden="true" />
              {models.length === 0 ? 'No voices yet.'
                : 'No matches.'}
            </li>
          ) : rows.map((model) => {
            const { name } = model;
            const size = model.size_bytes ?? model.size ?? null;
            const details = model.details || null;
            const isActive = activeModel === name;
            // The server prepares a voice completely (weights, index, warm-up) before it is
            // active; until then it shows as loading, also when restored at start-up.
            const isOperating = operatingOn === name || loadingModel === name;
            return (
              <li key={name} className={`rounded-lg border px-4 py-3 transition-colors ${
                isActive ? 'border-ok-line bg-ok-bg' : 'border-line bg-raised hover:border-line-hover'}`}>
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      {isActive ? <span className="h-2 w-2 shrink-0 rounded-full bg-ok-solid" aria-hidden="true" /> : null}
                      <span className="truncate text-sm font-semibold text-fg" title={name}>{name}</span>
                    </div>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <MetaBadge>{getModelType(name)}</MetaBadge>
                      {size != null && <MetaBadge>{formatBytes(size)}</MetaBadge>}
                      {model.has_index && <MetaBadge tone="emerald">Index</MetaBadge>}
                      {details?.version && <MetaBadge>{details.version}</MetaBadge>}
                      {details?.target_sample_rate && (
                        <MetaBadge>{(details.target_sample_rate / 1000).toFixed(0)} kHz</MetaBadge>
                      )}
                      {details?.realtime_index === false && <MetaBadge>Realtime index off</MetaBadge>}
                    </div>
                  </div>
                  {/* Same width in every state, so the row never shifts. */}
                  <div className="flex shrink-0 flex-wrap items-center gap-2">
                    {isActive ? (
                      <Button variant="success" onClick={handleDeactivate} loading={isOperating} className="w-28"
                        title="Click to deactivate">
                        Active
                      </Button>
                    ) : (
                      <Button onClick={() => handleActivate(name)} loading={isOperating} className="w-28"
                        title={isOperating ? 'Loading the voice and warming it up' : 'Use this voice'}>
                        {isOperating ? 'Loading…' : 'Activate'}
                      </Button>
                    )}
                    <Button href={modelDownloadUrl(name)} download icon={Download} className="w-28"
                      aria-label={`Download ${name} with its index`} title="Download the voice (.pth and .index in a .zip)">
                      Download
                    </Button>
                    <Button icon={Trash2} variant={deleteConfirm === name ? 'danger' : 'secondary'} className="w-28"
                      onClick={() => handleDelete(name)} disabled={isOperating}
                      aria-label={deleteConfirm === name ? `Confirm deleting ${name}` : `Delete ${name}`}
                      title={deleteConfirm === name ? 'Click again to delete permanently' : undefined}>
                      {deleteConfirm === name ? 'Confirm' : 'Delete'}
                    </Button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}

export default memo(ModelManager);
