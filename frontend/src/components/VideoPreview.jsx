import React, { memo } from 'react';

/**
 * The cloned face, as the call will see it: drawn by the video pipeline on
 * this canvas, already delayed to match the converted voice. Fills the stage
 * (the edges are cropped, as in a video call).
 */
function VideoPreview({ canvasRef, fps }) {
  return (
    <div className="absolute inset-0 bg-black">
      <canvas ref={canvasRef} className="h-full w-full object-cover" aria-label="Cloned video" />
      <div className="absolute right-2 top-2 rounded-sm bg-black/55 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-white/80">
        {fps} fps
      </div>
    </div>
  );
}

export default memo(VideoPreview);
