import React, { memo } from 'react';
import { Camera, ScanFace, Square, UserRound, Video, Webcam } from 'lucide-react';
import Button from './Button';
import Picker from './Picker';
import { faceThumbnailUrl } from '../lib/api';

const TONE = { danger: 'text-danger-fg', warn: 'text-warn-fg', muted: 'text-fg-subtle' };

/** The face clone's state in a few words, or null when there is nothing to say. */
export function videoNote(video, face) {
  const stats = video.stats;
  if (video.error) return { tone: 'danger', text: video.error };
  if (face.downloading) return { tone: 'muted', text: 'Downloading the face models…' };
  if (!face.ready) return { tone: 'muted', text: 'Face models not downloaded' };
  if (video.running && stats?.face === false) return { tone: 'warn', text: 'No face in view' };
  return null;
}

function FaceChoice({ selected, label, thumb, onClick, disabled }) {
  return (
    <button type="button" role="radio" aria-checked={selected} aria-label={label} title={label} disabled={disabled}
      onClick={onClick}
      className={`relative h-9 w-9 shrink-0 overflow-hidden rounded-md border-2 transition-colors disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)] ${
        selected ? 'border-video-solid' : 'border-transparent hover:border-line-hover'}`}>
      {thumb ? <img src={thumb} alt="" className="h-full w-full object-cover" />
        : <span className="flex h-full w-full items-center justify-center bg-sunken text-fg-subtle">
          <UserRound className="h-4 w-4" aria-hidden="true" />
        </span>}
    </button>
  );
}

/**
 * Live › the stage's header: the face to wear (thumbnails, one click), start
 * or stop the camera, which camera, and the virtual camera for calls.
 */
function VideoControls({ video, devices, face, onSetUp, onFaceChange, virtualCamera = null, onSetUpCalls }) {
  const { status, ready, activeFace } = face;
  const faces = status?.faces || [];
  const note = videoNote(video, face);
  const cameras = [{ value: '', label: 'Default camera' },
    ...(devices.cameraDevices || []).map((device) => ({
      value: device.deviceId, label: device.label || `Camera (${device.deviceId.slice(0, 8)}…)`,
    }))];
  // Without the virtual camera installed the switch leads to Settings › Calls.
  const camMissing = virtualCamera && !virtualCamera.installed;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex items-center gap-2">
        <ScanFace className="h-[18px] w-[18px] shrink-0 text-video-fg" aria-hidden="true" />
        <h2 id="live-face-title" className="text-sm font-semibold text-fg">Face</h2>
      </div>
      {ready ? (
        <div role="radiogroup" aria-labelledby="live-face-title"
          className="flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto px-1 py-0.5">
          <FaceChoice selected={!activeFace} label="My own face" onClick={() => onFaceChange(null)} disabled={face.busy} />
          {faces.map((item) => (
            <FaceChoice key={item.id} selected={activeFace?.id === item.id} label={item.name}
              thumb={faceThumbnailUrl(item.id)} onClick={() => onFaceChange(item.id)} disabled={face.busy} />
          ))}
        </div>
      ) : (
        <p className={`min-w-0 flex-1 truncate text-xs ${note ? TONE[note.tone] : ''}`} role="status">{note?.text}</p>
      )}
      <div className="flex items-center gap-2">
        {!ready ? (
          <Button size="md" icon={ScanFace} onClick={onSetUp} className="text-video-fg">Set up</Button>
        ) : video.running ? (
          <Button variant="danger" size="md" icon={Square} onClick={video.stop} className="w-28">Stop</Button>
        ) : (
          <Button size="md" icon={Video} onClick={video.start} loading={video.starting}
            className="w-28 border-video-line text-video-fg" title="V">
            {video.starting ? 'Starting…' : 'Camera'}
          </Button>
        )}
        <Picker label="Camera" icon={Camera} accent="face" value={video.cameraId} options={cameras}
          onChange={video.setCameraId} disabled={video.running} className="w-44 max-sm:hidden" />
        <Button size="md" icon={Webcam} selected={video.virtualCamera && !camMissing} role="switch"
          aria-checked={Boolean(video.virtualCamera && !camMissing)} aria-label="Virtual camera"
          title={camMissing ? 'Set up the virtual camera in Settings' : 'Virtual camera for calls'}
          onClick={() => (camMissing ? onSetUpCalls?.() : video.setVirtualCamera(!video.virtualCamera))}
          className={video.virtualCamera && !camMissing ? 'text-video-fg' : ''} />
      </div>
    </div>
  );
}

export default memo(VideoControls);
