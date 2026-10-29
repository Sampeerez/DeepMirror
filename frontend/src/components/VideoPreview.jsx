import React, { memo } from 'react';
import { ScanFace } from 'lucide-react';

/**
 * The cloned face, as the call will see it: drawn by the video pipeline on
 * this canvas, already delayed to match the converted voice.
 */
function VideoPreview({ canvasRef, fps, faceName }) {
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-md border border-video-line bg-black">
      <canvas ref={canvasRef} className="h-full w-full object-contain" aria-label="Cloned video preview" />
      <div className="absolute left-2 top-1.5 flex items-center gap-1.5 rounded-sm bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-video-fg">
        <ScanFace className="h-3 w-3" aria-hidden="true" />
        {faceName || 'Own face'}
      </div>
      <div className="absolute right-2 top-1.5 rounded-sm bg-black/55 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-white/80">
        {fps} fps
      </div>
    </div>
  );
}

export default memo(VideoPreview);
