// Pure helpers for the voice training page.

export const ACTIVE_STATES = new Set(['queued', 'running']);

export const STAGE_LABELS = {
  assets: 'Downloading base models',
  prepare: 'Cleaning and slicing audio',
  features: 'Extracting voice features',
  index: 'Building the retrieval index',
  train: 'Training',
  done: 'Finished',
};

export const STATE_LABELS = {
  queued: 'Queued',
  running: 'Training',
  done: 'Ready',
  stopped: 'Stopped',
  interrupted: 'Interrupted',
  error: 'Failed',
};

/** "1 h 05 min", "4 min 10 s", "12 s". */
export function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours) return `${hours} h ${String(minutes).padStart(2, '0')} min`;
  if (minutes) return `${minutes} min ${String(rest).padStart(2, '0')} s`;
  return `${rest} s`;
}

/** How good a dataset of this length is likely to be. */
export function datasetQuality(seconds, info) {
  const minimum = info?.min_audio_seconds ?? 60;
  const [low, high] = info?.recommended_audio_seconds ?? [600, 1800];
  const maximum = info?.max_audio_seconds ?? 3600;
  if (!seconds) return { tone: 'muted', text: 'Add clean recordings of one voice.' };
  if (seconds < minimum) return { tone: 'danger', text: `Too short: at least ${formatDuration(minimum)} is needed.` };
  if (seconds > maximum) return { tone: 'danger', text: `Too long: use at most ${formatDuration(maximum)}.` };
  if (seconds < low) return { tone: 'warn', text: `Usable. ${formatDuration(low)}–${formatDuration(high)} gives the most faithful voice.` };
  return { tone: 'ok', text: 'Great length for a faithful voice.' };
}

export function isActiveJob(job) {
  return ACTIVE_STATES.has(job?.state);
}

export function canResume(job) {
  return ['stopped', 'interrupted', 'error', 'done'].includes(job?.state);
}

/** Model file name a job exports to. */
export function jobModelFile(job) {
  return job?.model || (job?.name ? `${job.name}.pth` : null);
}

/** Read an audio file's duration in the browser (metadata only, no decoding). */
export function readAudioDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const done = (value) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => done(Number.isFinite(audio.duration) ? audio.duration : null);
    audio.onerror = () => done(null);
    audio.src = url;
  });
}
