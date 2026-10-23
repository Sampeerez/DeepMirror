import { describe, expect, it } from 'vitest';
import { canResume, datasetQuality, formatDuration, isActiveJob, jobModelFile } from './training';

const INFO = { min_audio_seconds: 60, recommended_audio_seconds: [600, 1800], max_audio_seconds: 3600 };

describe('training helpers', () => {
  it('formats durations compactly', () => {
    expect(formatDuration(12)).toBe('12 s');
    expect(formatDuration(250)).toBe('4 min 10 s');
    expect(formatDuration(3900)).toBe('1 h 05 min');
  });

  it('grades dataset length', () => {
    expect(datasetQuality(0, INFO).tone).toBe('muted');
    expect(datasetQuality(30, INFO).tone).toBe('danger');
    expect(datasetQuality(300, INFO).tone).toBe('warn');
    expect(datasetQuality(900, INFO).tone).toBe('ok');
    expect(datasetQuality(4000, INFO).tone).toBe('danger');
  });

  it('knows which jobs run, resume and what they export', () => {
    expect(isActiveJob({ state: 'running' })).toBe(true);
    expect(canResume({ state: 'interrupted' })).toBe(true);
    expect(canResume({ state: 'running' })).toBe(false);
    expect(jobModelFile({ name: 'voz' })).toBe('voz.pth');
  });
});
