import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Copy, Download, Pause, Play, Trash2 } from 'lucide-react';
import { fetchLogs, fetchTrainingJobs, fetchTrainingLog } from '../lib/api';

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
const LEVEL_CLASS = {
  DEBUG: 'text-fg-faint',
  INFO: 'text-fg-secondary',
  WARNING: 'text-warn-fg',
  ERROR: 'text-danger-fg',
  CRITICAL: 'text-danger-fg font-semibold',
};
const MAX_LINES = 3000;

function logsSocketUrl() {
  return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/logs`;
}

function formatTime(seconds) {
  const date = new Date(seconds * 1000);
  return date.toLocaleTimeString([], { hour12: false }) + `.${String(date.getMilliseconds()).padStart(3, '0')}`;
}

/** What the server prints (it runs without a console window), live, plus each training run's log. */
function LogsPanel({ active }) {
  const [source, setSource] = useState('server');
  const [jobs, setJobs] = useState([]);
  const [lines, setLines] = useState([]);
  const [minimum, setMinimum] = useState('INFO');
  const [query, setQuery] = useState('');
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState(null);
  const pausedRef = useRef(paused);
  const listRef = useRef(null);
  useEffect(() => { pausedRef.current = paused; }, [paused]);

  // Server log: the backlog, then live lines over a WebSocket.
  useEffect(() => {
    if (!active || source !== 'server') return undefined;
    let socket = null;
    let closed = false;
    fetchLogs(0)
      .then((body) => setLines(body.lines.slice(-MAX_LINES)))
      .catch((err) => setError(err.message));
    const open = () => {
      socket = new WebSocket(logsSocketUrl());
      socket.onmessage = (event) => {
        if (pausedRef.current) return;
        try {
          const { lines: fresh } = JSON.parse(event.data);
          setLines((previous) => [...previous, ...fresh].slice(-MAX_LINES));
        } catch { /* ignore malformed */ }
      };
      socket.onclose = () => { if (!closed) setTimeout(open, 2000); };
    };
    open();
    return () => { closed = true; socket?.close(); };
  }, [active, source]);

  // Training logs: the worker's file, refreshed every couple of seconds.
  useEffect(() => {
    if (!active) return undefined;
    fetchTrainingJobs().then(setJobs).catch(() => {});
    if (source === 'server') return undefined;
    let cancelled = false;
    const load = () => fetchTrainingLog(source)
      .then((body) => {
        if (cancelled || pausedRef.current) return;
        setLines(body.lines.map((message, index) => ({ id: index, time: null, level: 'INFO', logger: 'training', message })));
      })
      .catch((err) => setError(err.message));
    load();
    const timer = setInterval(load, 2000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active, source]);

  const visible = useMemo(() => {
    const threshold = LEVELS.indexOf(minimum);
    const needle = query.trim().toLowerCase();
    return lines.filter((line) => LEVELS.indexOf(line.level) >= threshold
      && (!needle || `${line.logger} ${line.message}`.toLowerCase().includes(needle)));
  }, [lines, minimum, query]);

  // The box has a fixed size and scrolls inside. It follows the newest line only
  // while you are at the bottom; scroll up to read and it stays put, counting new lines.
  const [following, setFollowing] = useState(true);
  const [unseen, setUnseen] = useState(0);
  const seenCountRef = useRef(0);
  const onScroll = () => {
    const list = listRef.current;
    if (!list) return;
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 24;
    setFollowing(atBottom);
    if (atBottom) setUnseen(0);
  };
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    if (following && !paused) {
      list.scrollTop = list.scrollHeight;
    } else {
      setUnseen((count) => count + Math.max(0, visible.length - seenCountRef.current));
    }
    seenCountRef.current = visible.length;
  }, [visible, following, paused]);
  const jumpToLatest = () => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
    setFollowing(true);
    setUnseen(0);
  };

  const asText = useCallback(() => visible.map((line) => (line.time
    ? `${formatTime(line.time)} [${line.level}] ${line.logger}: ${line.message}` : line.message)).join('\n'), [visible]);

  const download = () => {
    const url = URL.createObjectURL(new Blob([asText()], { type: 'text/plain' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = source === 'server' ? 'openvoicechanger-server.log' : `training-${source}.log`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="panel flex h-[calc(100dvh-14rem)] min-h-[420px] flex-col p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="panel-kicker">Logs</p>
          <h2 className="panel-title">What the studio is doing</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select aria-label="Log source" value={source} onChange={(event) => { setLines([]); setSource(event.target.value); }}
            className="native-select-safe rounded-md border border-line-strong bg-input px-2 py-1.5 text-xs text-fg">
            <option value="server">Server</option>
            {jobs.map((job) => <option key={job.id} value={job.id}>Training: {job.name}</option>)}
          </select>
          {source === 'server' ? (
            <select aria-label="Minimum level" value={minimum} onChange={(event) => setMinimum(event.target.value)}
              className="native-select-safe rounded-md border border-line-strong bg-input px-2 py-1.5 text-xs text-fg">
              {LEVELS.slice(0, 4).map((level) => <option key={level} value={level}>{level.toLowerCase()} and above</option>)}
            </select>
          ) : null}
          <input aria-label="Search the log" value={query} onChange={(event) => setQuery(event.target.value)}
            placeholder="Search" className="w-36 rounded-md border border-line-strong bg-input px-2 py-1.5 text-xs text-fg" />
          <button type="button" className="chip-button inline-flex items-center gap-1.5" onClick={() => setPaused((p) => !p)}>
            {paused ? <Play className="h-3.5 w-3.5" aria-hidden="true" /> : <Pause className="h-3.5 w-3.5" aria-hidden="true" />}
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button type="button" className="chip-button inline-flex items-center gap-1.5"
            onClick={() => navigator.clipboard?.writeText(asText())}>
            <Copy className="h-3.5 w-3.5" aria-hidden="true" /> Copy
          </button>
          <button type="button" className="chip-button inline-flex items-center gap-1.5" onClick={download}>
            <Download className="h-3.5 w-3.5" aria-hidden="true" /> Download
          </button>
          <button type="button" className="chip-button inline-flex items-center gap-1.5" onClick={() => setLines([])}
            aria-label="Clear the view">
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>
      {error ? <p className="mt-3 text-xs text-danger-fg">{error}</p> : null}
      <div className="relative mt-4 min-h-0 flex-1">
      <div ref={listRef} role="log" aria-live="off" onScroll={onScroll}
        className="h-full overflow-y-auto overscroll-contain rounded-md border border-line bg-raised p-3 font-mono text-[11px] leading-5">
        {visible.length === 0 ? <p className="text-fg-faint">Nothing to show yet.</p> : visible.map((line) => (
          <div key={`${line.id}-${line.time}`} className={`whitespace-pre-wrap break-words ${LEVEL_CLASS[line.level] || ''}`}>
            {line.time ? <span className="text-fg-faint">{formatTime(line.time)} </span> : null}
            {line.time ? <span className="text-fg-subtle">{line.logger}: </span> : null}
            {line.message}
          </div>
        ))}
      </div>
      {!following ? (
        <button type="button" onClick={jumpToLatest}
          className="chip-button absolute bottom-3 right-5 shadow-md">
          Jump to latest{unseen ? ` (${unseen} new)` : ''}
        </button>
      ) : null}
      </div>
    </section>
  );
}

export default memo(LogsPanel);
