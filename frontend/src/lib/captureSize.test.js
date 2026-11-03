import { describe, expect, it } from 'vitest';
import { captureSize } from './captureSize';

describe('captureSize', () => {
  it('keeps the shape of any camera', () => {
    expect(captureSize(1920, 1080, 1280)).toEqual({ width: 1280, height: 720 });
    expect(captureSize(1080, 1920, 1280)).toEqual({ width: 720, height: 1280 });
    expect(captureSize(1600, 1200, 1280)).toEqual({ width: 1280, height: 960 });
    expect(captureSize(1000, 1000, 1280)).toEqual({ width: 1000, height: 1000 });
  });

  it('never enlarges a small camera', () => {
    expect(captureSize(640, 480, 1280)).toEqual({ width: 640, height: 480 });
  });

  it('gives even sizes and survives a camera not started yet', () => {
    expect(captureSize(641, 359, 1280)).toEqual({ width: 642, height: 360 });
    expect(captureSize(0, 0, 1280)).toEqual({ width: 2, height: 2 });
  });
});
