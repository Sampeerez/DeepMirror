import React, { memo } from 'react';
import { Camera, ScanFace, Square, Video, Webcam } from 'lucide-react';
import Button from './Button';

/** One line on the face clone's state, for the Live bar's status line. */
export function videoNote(video, face) {
  const { ready, activeFace } = face;
  const stats = video.stats;
  if (video.error) return { tone: 'danger', text: video.error };
  if (face.downloading) return { tone: 'muted', text: 'Downloading the face models…' };
  if (!ready) return { tone: 'muted', text: 'Video: download the face models once (Set up video).' };
  if (!activeFace) return { tone: 'muted', text: 'Video: add a face in Video › Faces.' };
  if (!video.running) return { tone: 'muted', text: `Video: wearing ${activeFace.name}.` };
  if (stats?.face === false) return { tone: 'warn', text: 'Video: no face in view.' };
  return {
    tone: 'muted',
    text: `Video: ${video.shownFps} fps · ${stats?.keys_per_second ?? 0} swaps/s${stats?.camera ? ` · ${stats.camera}` : ''}`,
  };
}

/**
 * The Live bar's video row, under the voice row and on the same columns:
 * camera on/off, which camera, which face, and the virtual camera.
 */
function VideoControls({ video, devices, face, onSetUp, onFaceChange }) {
  const { status, ready, activeFace } = face;
  const faces = status?.faces || [];

  return (
    <div className="mt-3 grid gap-3 border-t border-line pt-3 md:grid-cols-2 xl:grid-cols-[10rem_minmax(0,1fr)_minmax(0,1fr)_auto] xl:items-center">
      {!ready ? (
        <Button size="md" icon={ScanFace} onClick={onSetUp} className="w-full text-video-fg">Set up video</Button>
      ) : video.running ? (
        <Button variant="danger" size="md" icon={Square} onClick={video.stop} className="w-full">Stop video</Button>
      ) : (
        <Button size="md" icon={Video} onClick={video.start} loading={video.starting} disabled={!activeFace}
          className="w-full border-video-line text-video-fg" title="Shortcut: V">
          {video.starting ? 'Starting…' : 'Start video'}
        </Button>
      )}

      <div className="flex items-center gap-2 md:justify-end xl:order-last">
        <Button size="md" icon={Webcam} selected={video.virtualCamera}
          onClick={() => video.setVirtualCamera(!video.virtualCamera)} className={`w-[16.5rem] ${video.virtualCamera ? 'text-video-fg' : ''}`}
          title="Sends the cloned face to OBS Virtual Camera (install OBS Studio once): pick it as the camera in Discord, Meet, Teams…">
          {video.virtualCamera ? 'Virtual camera: on' : 'Virtual camera: off'}
        </Button>
      </div>

      <label htmlFor="video-camera" className="flex min-w-0 items-center gap-2">
        <Camera className="h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />
        <span className="sr-only">Camera</span>
        <select id="video-camera" value={video.cameraId} onChange={(event) => video.setCameraId(event.target.value)}
          disabled={video.running} className="native-select-safe field">
          <option value="">Default camera</option>
          {(devices.cameraDevices || []).map((device) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera (${device.deviceId.slice(0, 8)}…)`}
            </option>
          ))}
        </select>
      </label>

      <label htmlFor="video-face" className="flex min-w-0 items-center gap-2">
        <ScanFace className="h-4 w-4 shrink-0 text-video-fg" aria-hidden="true" />
        <span className="sr-only">Face</span>
        <select id="video-face" value={activeFace?.id || ''} disabled={!ready || face.busy}
          onChange={(event) => onFaceChange(event.target.value || null)} className="native-select-safe field">
          <option value="">{faces.length ? 'My own face (no swap)' : 'No faces yet'}</option>
          {faces.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </label>
    </div>
  );
}

export default memo(VideoControls);
