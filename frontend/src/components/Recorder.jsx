import React, { memo, useEffect, useRef, useState } from 'react';
import { Circle, Download, Square, Trash2, Check, Pencil, RefreshCw, X } from 'lucide-react';
import { formatBytes, formatDuration } from '../lib/format';
import Button from './Button';

// The elapsed counter ticks at animation rate. It subscribes to the pipeline's
// meter store and writes the text itself, so a running recording never
// re-renders the rest of the studio.
function RecordTimer({ meters }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!meters?.subscribe) return undefined;
    let shown = null;
    return meters.subscribe((value) => {
      const text = formatDuration(value?.recordSeconds ?? 0);
      if (text !== shown && ref.current) {
        ref.current.textContent = text;
        shown = text;
      }
    });
  }, [meters]);

  return <span ref={ref}>0:00</span>;
}

// One saved take: name, length, player and actions on two compact rows.
function TakeRow({ take, library }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(take.name);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const renameRef = useRef(null);
  const deleteRef = useRef(null);
  const finishEditing = () => {
    setEditing(false);
    requestAnimationFrame(() => renameRef.current?.focus());
  };
  return (
    <li className="min-w-0 py-2.5">
      {editing ? (
        <form className="flex gap-2" onSubmit={(event) => {
          event.preventDefault();
          if (name.trim()) { library.renameTake(take, name); finishEditing(); }
        }}>
          <input autoFocus aria-label="Take name" maxLength={120} value={name} required
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Escape') finishEditing(); }}
            className="field h-8 flex-1" />
          <Button type="submit" icon={Check} aria-label="Save name" title="Save name" disabled={!name.trim()} />
          <Button icon={X} aria-label="Cancel rename" title="Cancel rename" onClick={finishEditing} />
        </form>
      ) : (
        <div className="flex items-baseline justify-between gap-3">
          <p className="min-w-0 truncate text-xs font-medium text-fg-secondary" title={take.name}>{take.name}</p>
          <p className="shrink-0 font-mono text-[11px] text-fg-faint">
            {formatDuration(take.seconds)} · {formatBytes(take.size)}{take.pending ? ' · saving…' : ''}
          </p>
        </div>
      )}
      {!take.persisted && !take.pending ? (
        <p role="status" className="mt-1 text-[11px] text-warn-fg">Not saved: download it or retry.</p>
      ) : null}
      <div className="mt-1.5 flex items-center gap-1">
        <audio controls preload="none" src={take.url} aria-label={`Play ${take.name}`} className="h-8 min-w-0 flex-1" />
        <Button href={take.url} download={take.fileName} onClick={() => library.markDownloaded(take.id)} icon={Download}
          variant="ghost" title="Download WAV" aria-label={`Download ${take.name}`} />
        <Button ref={renameRef} icon={Pencil} variant="ghost" disabled={take.pending} title="Rename take"
          aria-label={`Rename ${take.name}`} onClick={() => { setName(take.name); setEditing(true); }} />
        {!take.persisted ? (
          <Button icon={RefreshCw} variant="ghost" disabled={take.pending} title="Retry saving"
            aria-label={`Retry saving ${take.name}`} onClick={() => library.retrySave(take)} />
        ) : null}
        <Button ref={deleteRef} icon={Trash2} variant="ghost" disabled={take.pending} title="Delete take"
          aria-label={`Delete ${take.name}`} onClick={() => setConfirmDelete(true)} />
      </div>
      {confirmDelete ? (
        <div className="mt-1.5 flex items-center justify-end gap-2 text-[11px]">
          <span className="text-fg-secondary">Delete permanently?</span>
          <Button variant="danger" disabled={take.pending} onClick={() => {
            setConfirmDelete(false); void library.deleteTake(take);
          }}>Delete</Button>
          <Button onClick={() => { setConfirmDelete(false); deleteRef.current?.focus(); }}>Cancel</Button>
        </div>
      ) : null}
    </li>
  );
}

/** Records the converted output; the takes list scrolls inside the Monitor card. */
function Recorder({ pipeline }) {
  const { isRunning, isRecording, library, recordNotice, meters } = pipeline;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between gap-3">
        <p className="min-w-0 truncate text-xs font-semibold text-fg-secondary">
          Recorder · {library.takes.length} take{library.takes.length === 1 ? '' : 's'}
        </p>
        <div className="flex shrink-0 items-center gap-3">
          {isRecording ? (
            <span className="flex items-center gap-2 font-mono text-xs tabular-nums text-danger-fg">
              <span className="rec-blink inline-block h-2 w-2 rounded-full bg-danger-solid" />
              <RecordTimer meters={meters} />
            </span>
          ) : null}
          {isRecording ? (
            <Button variant="danger" icon={Square} onClick={() => pipeline.stopRecording()} className="w-24"
              title="Shortcut: R">Stop</Button>
          ) : (
            <Button variant="danger" icon={Circle} onClick={() => pipeline.startRecording()} disabled={!isRunning}
              className="w-24" title={isRunning ? 'Shortcut: R' : 'Start first'}>
              Record
            </Button>
          )}
        </div>
      </div>

      {recordNotice ? (
        <p role="status" className={`mt-2 break-words text-[11px] ${recordNotice.tone === 'error' ? 'text-danger-fg' : 'text-warn-fg'}`}>
          {recordNotice.message}
        </p>
      ) : null}
      {library.error ? <p role="alert" className="mt-2 break-words text-[11px] text-warn-fg">{library.error}</p> : null}

      <ul className="mt-2 max-h-72 min-h-0 flex-1 divide-y divide-line overflow-y-auto pr-1 lg:max-h-none">
        {library.loading ? <li role="status" className="py-2 text-[11px] text-fg-subtle">Loading saved takes…</li> : null}
        {!library.loading && !library.takes.length ? (
          <li className="py-2 text-[11px] text-fg-faint">No takes yet.</li>
        ) : null}
        {library.takes.map((take) => <TakeRow key={take.id} take={take} library={library} />)}
      </ul>
    </div>
  );
}

export default memo(Recorder);
