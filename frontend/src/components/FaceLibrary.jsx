import React, { memo, useRef, useState } from 'react';
import { Box, Check, Download, ImagePlus, ScanFace, Trash2, UserRound } from 'lucide-react';
import Button from './Button';
import { faceThumbnailUrl } from '../lib/api';
import { formatBytes, headLabel } from '../lib/format';

const ACCEPTED = 'image/jpeg,image/png,image/webp';
const MAX_PHOTOS = 12;

function ModelDownload({ face }) {
  const assets = face.status?.assets || [];
  const total = assets.reduce((sum, asset) => sum + asset.bytes, 0);
  const received = assets.reduce((sum, asset) => sum + (asset.state === 'ready' ? asset.bytes : asset.received || 0), 0);
  const failed = assets.filter((asset) => asset.state === 'error');
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <p className="text-sm text-fg-secondary">
        The face clone needs its models once ({formatBytes(total)}). They run on your computer; nothing is uploaded.
      </p>
      <div className="mt-4">
        <Button variant="primary" size="md" icon={Download} onClick={face.prepare}
          loading={face.downloading} disabled={face.busy}>
          {face.downloading ? `Downloading ${total ? Math.round((received / total) * 100) : 0}%` : 'Download the face models'}
        </Button>
      </div>
      <ul className="mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
        {assets.map((asset) => {
          const percent = asset.state === 'ready' ? 100 : Math.round(((asset.received || 0) / asset.bytes) * 100);
          return (
            <li key={asset.id} className="text-xs">
              <div className="flex items-center justify-between gap-3">
                <span className="min-w-0 truncate text-fg-secondary">{asset.label}</span>
                <span className="shrink-0 font-mono text-fg-faint">
                  {asset.state === 'ready' ? 'Ready' : asset.state === 'error' ? 'Error' : `${percent}%`} · {formatBytes(asset.bytes)}
                </span>
              </div>
              <div className="mt-1 h-1 overflow-hidden rounded-full bg-meter-track">
                <div className={`h-full ${asset.state === 'error' ? 'bg-danger-solid' : 'bg-video-solid'}`}
                  style={{ width: `${percent}%` }} />
              </div>
            </li>
          );
        })}
      </ul>
      {failed.length ? (
        <p className="mt-3 break-words text-xs text-danger-fg" role="alert">{failed[0].error}</p>
      ) : null}
    </div>
  );
}

function AddFace({ face }) {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [name, setName] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const pick = (list) => setFiles((previous) => [...previous, ...Array.from(list || [])
    .filter((file) => file.type.startsWith('image/'))].slice(0, MAX_PHOTOS));

  const submit = async () => {
    if (!files.length) return;
    if (await face.add(files, name.trim())) {
      setFiles([]);
      setName('');
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        role="button"
        tabIndex={0}
        aria-label="Add photos of the face"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inputRef.current?.click(); }
        }}
        onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => { event.preventDefault(); setDragActive(false); pick(event.dataTransfer.files); }}
        className={`flex cursor-pointer items-center justify-center gap-3 rounded-lg border border-dashed px-4 py-4 transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
          dragActive ? 'border-video-line bg-video-bg' : 'border-line-strong bg-raised hover:border-line-hover hover:bg-control'}`}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED} multiple className="hidden"
          onChange={(event) => { pick(event.target.files); event.target.value = ''; }} />
        <ImagePlus className="h-6 w-6 shrink-0 text-video-fg" strokeWidth={1.5} aria-hidden="true" />
        <div className="min-w-0 text-left">
          <p className="text-sm font-medium text-fg-secondary">
            {files.length ? `${files.length} photo${files.length === 1 ? '' : 's'} chosen` : 'Drop photos or click to choose'}
          </p>
          <p className="mt-0.5 text-xs text-fg-faint">1–{MAX_PHOTOS} photos · the sharpest one, front-facing, makes the head</p>
        </div>
      </div>
      {files.length ? (
        <div className="grid grid-cols-6 gap-1.5">
          {files.map((file, index) => (
            <img key={`${file.name}-${index}`} src={URL.createObjectURL(file)} alt=""
              onLoad={(event) => URL.revokeObjectURL(event.currentTarget.src)}
              className="aspect-square w-full rounded-sm border border-line object-cover" />
          ))}
        </div>
      ) : null}
      <div>
        <label htmlFor="face-name" className="block text-xs font-medium text-fg-muted">Name</label>
        <input id="face-name" className="field mt-1.5" value={name} maxLength={60} placeholder="My character"
          onChange={(event) => setName(event.target.value)} />
      </div>
      <p className="text-xs text-fg-subtle">Only use faces you have permission to use.</p>
      <div className="mt-auto border-t border-line pt-3">
        <Button variant="primary" size="md" icon={Check} onClick={submit} disabled={!files.length}
          loading={face.busy} className="w-40">Add face</Button>
      </div>
    </div>
  );
}

function FaceTile({ item, wearing, onWear, onDelete, onBuild, busy }) {
  const head = item?.head3d;
  const building = ['building', 'queued'].includes(head?.state);
  return (
    <li className={`flex flex-col overflow-hidden rounded-lg border ${wearing ? 'border-video-line bg-video-bg' : 'border-line-strong bg-raised'}`}>
      {item ? (
        <img src={faceThumbnailUrl(item.id)} alt="" className="aspect-square w-full object-cover" />
      ) : (
        <div className="flex aspect-square w-full items-center justify-center bg-sunken">
          <UserRound className="h-10 w-10 text-fg-faint" strokeWidth={1.5} aria-hidden="true" />
        </div>
      )}
      <div className="flex flex-1 flex-col gap-2 p-2.5">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-fg" title={item?.name}>{item ? item.name : 'My own face'}</p>
          <p className="text-[11px] text-fg-faint">
            {item ? `${item.photos} photo${item.photos === 1 ? '' : 's'}` : 'No swap'}
          </p>
          {onBuild && head ? (
            <p className={`truncate text-[11px] ${head.state === 'failed' ? 'text-danger-fg' : 'text-fg-faint'}`}
              title={head.error || undefined}>{headLabel(head)}</p>
          ) : null}
        </div>
        <div className="mt-auto flex items-center gap-1.5">
          <Button size="sm" icon={wearing ? Check : ScanFace} onClick={onWear} disabled={busy || wearing}
            selected={wearing} className={`min-w-0 flex-1 ${wearing ? 'text-video-fg' : ''}`}>
            {wearing ? 'Wearing' : 'Wear'}
          </Button>
          {onBuild && head ? (
            <Button size="sm" icon={Box} onClick={onBuild} disabled={busy || building} loading={building}
              aria-label={`Build the 3D head of ${item.name}`}
              title={head.state === 'ready' ? 'Rebuild the 3D head' : 'Build the 3D head'} />
          ) : null}
          {item ? (
            <Button size="sm" variant="danger" icon={Trash2} onClick={onDelete} disabled={busy}
              aria-label={`Delete ${item.name}`} title="Delete this face" />
          ) : null}
        </div>
      </div>
    </li>
  );
}

/** Face › Faces: the faces to wear, adding a new one, and the one-time model download. */
function FaceLibrary({ face }) {
  const faces = face.status?.faces || [];
  const active = face.status?.active_face || null;
  return (
    <div className="page-grid">
      <section className="panel flex min-h-0 flex-col p-4 lg:col-span-2" aria-labelledby="faces-title">
        <h2 id="faces-title" className="panel-title">Faces</h2>
        <ul className="mt-4 grid min-h-0 flex-1 auto-rows-min grid-cols-2 gap-3 overflow-y-auto pr-1 sm:grid-cols-3 xl:grid-cols-5">
          <FaceTile item={null} wearing={!active} onWear={() => face.activate(null)} busy={face.busy} />
          {faces.map((item) => (
            <FaceTile key={item.id} item={item} wearing={active === item.id} busy={face.busy}
              onWear={() => face.activate(item.id)} onDelete={() => face.remove(item.id)}
              onBuild={face.status?.head3d ? () => face.buildHead(item.id) : null} />
          ))}
        </ul>
        {face.error ? <p className="mt-3 break-words text-xs text-danger-fg" role="alert">{face.error}</p> : null}
      </section>

      <section className="panel flex min-h-0 flex-col p-4" aria-labelledby="faces-add-title">
        <h2 id="faces-add-title" className="panel-title">{face.ready ? 'Add a face' : 'Set up the face clone'}</h2>
        <div className="mt-4 flex min-h-0 flex-1 flex-col">
          {face.ready ? <AddFace face={face} /> : <ModelDownload face={face} />}
        </div>
      </section>
    </div>
  );
}

export default memo(FaceLibrary);
