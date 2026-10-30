import { API_BASE, DEFAULT_F0_METHOD } from './constants';

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/** Message for a request that never produced a response. */
function describeTransportError(err) {
  if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
    return 'The server did not respond in time.';
  }
  return 'Cannot reach the server. Is the backend running?';
}

/** JSON request. `timeoutMs` stops waiting on a hung backend; a caller's `signal` wins. */
async function request(url, { timeoutMs, signal, ...options } = {}) {
  const requestSignal = signal
    ?? (timeoutMs > 0 && typeof AbortSignal?.timeout === 'function'
      ? AbortSignal.timeout(timeoutMs)
      : undefined);

  let response;
  try {
    response = await fetch(`${API_BASE}${url}`, {
      ...options,
      signal: requestSignal,
      headers: {
        ...(options.body instanceof FormData
          ? {}
          : { 'Content-Type': 'application/json' }),
        ...options.headers,
      },
    });
  } catch (err) {
    if (signal) throw err;
    throw new ApiError(describeTransportError(err), 0, null);
  }

  if (!response.ok) {
    let data = null;
    try {
      data = await response.json();
    } catch {
      // Response may not be JSON
    }
    throw new ApiError(
      data?.detail || `Request failed: ${response.status} ${response.statusText}`,
      response.status,
      data
    );
  }

  const text = await response.text();
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function fetchModels() {
  return request('/models/');
}

/** POST a form with upload progress (0..100; 99 until the server answers). XHR, since fetch has no upload progress. */
function postFormWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}${url}`);
    xhr.responseType = 'text';

    if (typeof onProgress === 'function') {
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable || event.total === 0) return;
        onProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)));
      };
      xhr.upload.onload = () => onProgress(99);
    }

    xhr.onload = () => {
      let data = null;
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        // Response may not be JSON.
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        resolve(data ?? xhr.responseText ?? null);
        return;
      }
      reject(new ApiError(data?.detail || `Upload failed: ${xhr.status} ${xhr.statusText}`, xhr.status, data));
    };
    xhr.onerror = () => reject(new ApiError('Upload failed: network error', 0, null));
    xhr.onabort = () => reject(new ApiError('Upload cancelled', 0, null));
    xhr.ontimeout = () => reject(new ApiError('Upload timed out', 0, null));
    xhr.send(formData);
  });
}

export function uploadModel(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);
  return postFormWithProgress('/models/upload', formData, onProgress);
}

// --- Voice training --------------------------------------------------------

export const fetchTrainingInfo = () => request('/training/info', { timeoutMs: 15000 });
export const fetchTrainingJobs = () => request('/training/jobs', { timeoutMs: 15000 });

export function createTrainingJob({
  name, files, sampleRate, epochs, batchSize, saveEvery, overwrite, base, quality, isolate,
}, onProgress) {
  const formData = new FormData();
  formData.append('name', name);
  for (const audio of files) formData.append('files', audio);
  formData.append('sample_rate', String(sampleRate));
  // No epochs: the server plans them (and the batch) from the audio and the GPU.
  if (epochs) formData.append('epochs', String(epochs));
  if (batchSize) formData.append('batch_size', String(batchSize));
  if (saveEvery) formData.append('save_every', String(saveEvery));
  if (overwrite) formData.append('overwrite', 'true');
  if (base) formData.append('base', base);
  if (quality) formData.append('quality', quality);
  if (isolate === false) formData.append('isolate', 'false');
  return postFormWithProgress('/training/jobs', formData, onProgress);
}

export const stopTrainingJob = (id) => request(`/training/jobs/${id}/stop`, { method: 'POST' });
export const resumeTrainingJob = (id, epochs) => request(`/training/jobs/${id}/resume`, {
  method: 'POST',
  body: JSON.stringify(epochs ? { epochs } : {}),
});
export const deleteTrainingJob = (id) => request(`/training/jobs/${id}`, { method: 'DELETE' });

export async function deleteModel(name) {
  return request(`/models/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
}

export async function activateModel(name) {
  return request(`/models/${encodeURIComponent(name)}/activate`, {
    method: 'POST',
  });
}

export async function deactivateModel() {
  return request('/models/deactivate', {
    method: 'POST',
  });
}

// Waits on the model lock while a checkpoint loads, so the periodic poll
// passes a timeout and the model bay (which runs after activation) does not.
export async function getActiveModel(options = {}) {
  return request('/models/active', options);
}

// Lock-free on the server; the bound only guards against a hung process so
// the WebSocket still gets connected.
export async function fetchConfig() {
  return request('/config', { timeoutMs: 15000 });
}

export async function fetchPresets() {
  return request('/presets/', { timeoutMs: 10000 });
}

export async function savePreset(name, settings) {
  return request('/presets/', {
    method: 'POST',
    body: JSON.stringify({ name, settings }),
  });
}

export async function deletePreset(presetId) {
  return request(`/presets/${encodeURIComponent(presetId)}`, {
    method: 'DELETE',
  });
}

/** Convert a file; resolves to a WAV Blob. No timeout: cancel with `signal`. */
export async function convertFile(file, {
  pitchShift = 0,
  formantShift = 0,
  f0Method = DEFAULT_F0_METHOD,
  effects = {},
  useModel = true,
  model = null,
  autoPitch = false,
  expressiveness,
  pitchCorrection,
  indexRate,
  filterRadius,
  rmsMixRate,
  protect,
  crepeHopLength,
  signal,
} = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('pitch_shift', String(pitchShift));
  formData.append('pitch_mode', autoPitch ? 'auto' : 'manual');
  formData.append('formant_shift', String(formantShift));
  formData.append('f0_method', f0Method);
  formData.append('effects', JSON.stringify(effects));
  formData.append('use_model', String(useModel));
  if (model) formData.append('model', model);

  // Omitted controls use the server default.
  for (const [field, value] of [
    ['index_rate', indexRate],
    ['filter_radius', filterRadius],
    ['rms_mix_rate', rmsMixRate],
    ['protect', protect],
    ['crepe_hop_length', crepeHopLength],
    ['pitch_expressiveness', expressiveness],
    ['pitch_correction', pitchCorrection],
  ]) {
    if (Number.isFinite(Number(value)) && value !== null && value !== '') {
      formData.append(field, String(value));
    }
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/convert/`, {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError(describeTransportError(err), 0, null);
  }

  if (!response.ok) {
    let data = null;
    try {
      data = await response.json();
    } catch {
      // Response may not be JSON
    }
    throw new ApiError(
      data?.detail || `Conversion failed: ${response.status} ${response.statusText}`,
      response.status,
      data
    );
  }

  return response.blob();
}

/** True when a rejection came from the caller's own AbortController. */
export function isAbortError(err) {
  return err?.name === 'AbortError';
}

// --- Virtual microphone (calls) and server logs -----------------------------

export const fetchVoiceProfile = () => request('/profile/', { timeoutMs: 15000 });
export function saveVoiceProfile(wavBlob, language) {
  const formData = new FormData();
  formData.append('file', wavBlob, 'my-voice.wav');
  formData.append('language', language);
  return request('/profile/', { method: 'POST', body: formData });
}
export const deleteVoiceProfile = () => request('/profile/', { method: 'DELETE' });
export function saveVoiceReference(modelName, file) {
  const formData = new FormData();
  formData.append('file', file);
  return request(`/profile/reference/${encodeURIComponent(modelName)}`, { method: 'POST', body: formData });
}
export const deleteVoiceReference = (modelName) => request(`/profile/reference/${encodeURIComponent(modelName)}`, {
  method: 'DELETE',
});

export const fetchVirtualMic = () => request('/virtual-mic');
export const installVirtualMic = () => request('/virtual-mic/install', { method: 'POST' });
/** {microphone, camera}: what calls need, installed or not. */
export const fetchCalls = () => request('/calls');
export const setupCalls = () => request('/calls/setup', { method: 'POST' });
export const fetchLogs = (after = 0, level = null) =>
  request(`/logs?after=${after}${level ? `&level=${encodeURIComponent(level)}` : ''}&limit=2000`);
export const fetchTrainingLog = (jobId, lines = 800) => request(`/training/jobs/${jobId}/log?lines=${lines}`);

/** Direct link to a voice as a .zip (its .pth plus its .index). */
export const modelDownloadUrl = (name) => `${API_BASE}/models/${encodeURIComponent(name)}/download`;

// --- Face clone -------------------------------------------------------------

export const fetchVideo = () => request('/video/', { timeoutMs: 15000 });
export const prepareVideo = () => request('/video/prepare', { method: 'POST' });
export const updateVideoSettings = (settings) => request('/video/settings', {
  method: 'PATCH', body: JSON.stringify(settings),
});
export function addFace(files, name) {
  const formData = new FormData();
  for (const file of files) formData.append('files', file);
  formData.append('name', name || '');
  return request('/video/faces', { method: 'POST', body: formData });
}
export const activateFace = (faceId) => request(`/video/faces/${encodeURIComponent(faceId || 'none')}/activate`, {
  method: 'POST',
});
export const deleteFace = (faceId) => request(`/video/faces/${encodeURIComponent(faceId)}`, { method: 'DELETE' });
export const faceThumbnailUrl = (faceId) => `${API_BASE}/video/faces/${encodeURIComponent(faceId)}/thumbnail`;

/** Start converting a video (face on every frame, voice on its sound); returns the job. */
export function startVideoConversion(file, { face = true, clone = 'hair', model = null, useModel = true, pitchShift = 0,
  autoPitch = false, formantShift = 0, f0Method, effects = {}, indexRate, protect, expressiveness } = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('face', typeof face === 'string' ? face : face ? 'on' : 'off');   // a face id, the live one, or none
  formData.append('clone', clone);
  formData.append('use_model', String(useModel));
  if (model) formData.append('model', model);
  formData.append('pitch_shift', String(pitchShift));
  formData.append('pitch_mode', autoPitch ? 'auto' : 'manual');
  formData.append('formant_shift', String(formantShift));
  if (f0Method) formData.append('f0_method', f0Method);
  formData.append('effects', JSON.stringify(effects));
  for (const [field, value] of [['index_rate', indexRate], ['protect', protect],
    ['pitch_expressiveness', expressiveness]]) {
    if (value !== undefined && value !== null) formData.append(field, String(value));
  }
  return request('/convert/video', { method: 'POST', body: formData });
}
export const fetchVideoConversion = (jobId) => request(`/convert/video/${encodeURIComponent(jobId)}`);
export const videoConversionUrl = (jobId) => `${API_BASE}/convert/video/${encodeURIComponent(jobId)}/file`;
