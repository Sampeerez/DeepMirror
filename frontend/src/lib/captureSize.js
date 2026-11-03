/** The size a camera's frame travels at: its own shape, with its longest side at most ``longest``, in even pixels. */
export function captureSize(width, height, longest) {
  const scale = Math.min(1, longest / Math.max(width, height, 1));
  const even = (value) => Math.max(2, Math.round((value * scale) / 2) * 2);
  return { width: even(width), height: even(height) };
}
