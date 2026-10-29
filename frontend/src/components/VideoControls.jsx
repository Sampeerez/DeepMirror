import React, { memo } from 'react';
import { Camera, ScanFace, Square, Video, Webcam } from 'lucide-react';
import Button from './Button';
import Picker from './Picker';
import { faceThumbnailUrl } from '../lib/api';

const TONE = { danger: 'text-danger-fg', warn: 'text-warn-fg', muted: 'text-fg-subtle' };

/** One line on the face clone's state, for the Live bar's status line. */
export function videoNote(video, face) {
  const { ready, activeFace } = face;
  const stats = video.stats;
  if (video.error) return { tone: 'danger', text: video.error };
  if (face.downloading) return { tone: 'muted', text: 'Downloading the face models…' };
  if (!ready) return { tone: 'muted', text: 'Download the face models once (Set up).' };
  if (!video.running) return { tone: 'muted', text: activeFace ? `Ready to wear ${activeFace.name}.` : 'Your own face.' };
  if (stats?.face === false) return { tone: 'warn', text: 'No face in view.' };
  return {
    tone: 'muted',
    text: `${video.shownFps} fps · ${stats?.keys_per_second ?? 0} swaps/s${stats?.camera ? ` · ${stats.camera}` : ''}`,
  };
}

/**
 * Live › Face: start/stop the camera, which camera, which face to wear, and
 * the virtual camera, with its own status line. One column wide.
 */
function VideoControls({ video, devices, face, onSetUp, onFaceChange }) {
  const { status, ready, activeFace } = face;
  const faces = status?.faces || [];
  const note = videoNote(video, face);
  const cameras = [{ value: '', label: 'Default camera' },
    ...(devices.cameraDevices || []).map((device) => ({
      value: device.deviceId, label: device.label || `Camera (${device.deviceId.slice(0, 8)}…)`,
    }))];
  const faceOptions = [{ value: '', label: 'My own face' },
    ...faces.map((item) => ({ value: item.id, label: item.name, thumb: faceThumbnailUrl(item.id) }))];

  return (
    <section className="panel flex flex-col p-4" aria-labelledby="live-face-title">
      <div className="flex h-8 items-center gap-3">
        <ScanFace className="h-5 w-5 shrink-0 text-video-fg" aria-hidden="true" />
        <h2 id="live-face-title" className="text-sm font-semibold text-fg">Face</h2>
        <p className={`min-w-0 flex-1 truncate text-[11px] ${TONE[note.tone]}`} role="status" aria-live="polite"
          title={note.text}>{note.text}</p>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-[9rem_minmax(0,1fr)]">
        {!ready ? (
          <Button size="md" icon={ScanFace} onClick={onSetUp} className="w-full text-video-fg">Set up</Button>
        ) : video.running ? (
          <Button variant="danger" size="md" icon={Square} onClick={video.stop} className="w-full">Stop</Button>
        ) : (
          <Button size="md" icon={Video} onClick={video.start} loading={video.starting}
            className="w-full border-video-line text-video-fg" title="Start the camera (shortcut: V)">
            {video.starting ? 'Starting…' : 'Start'}
          </Button>
        )}
        <Picker label="Camera" icon={Camera} accent="face" value={video.cameraId} options={cameras}
          onChange={video.setCameraId} disabled={video.running} />
        <Picker label="Wear" icon={ScanFace} accent="face" value={activeFace?.id || ''} options={faceOptions}
          onChange={(id) => onFaceChange(id || null)} disabled={!ready || face.busy}
          render={(option) => (
            <span className="flex min-w-0 flex-1 items-center gap-2">
              {option.thumb ? <img src={option.thumb} alt="" className="h-6 w-6 shrink-0 rounded-sm object-cover" />
                : <span className="h-6 w-6 shrink-0 rounded-sm bg-sunken" aria-hidden="true" />}
              <span className="min-w-0 truncate">{option.label}</span>
            </span>
          )} />
        <Button size="md" icon={Webcam} selected={video.virtualCamera} role="switch" aria-checked={video.virtualCamera}
          onClick={() => video.setVirtualCamera(!video.virtualCamera)}
          className={video.virtualCamera ? 'text-video-fg' : ''}
          title="Sends the face to OBS Virtual Camera (install OBS Studio once): pick it as the camera in Discord, Meet, Teams…">
          {video.virtualCamera ? 'Virtual cam: on' : 'Virtual cam'}
        </Button>
      </div>
    </section>
  );
}

export default memo(VideoControls);
