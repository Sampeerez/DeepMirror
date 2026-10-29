import { describe, expect, it } from 'vitest';
import {
  bestEpochSummary, canResume, datasetQuality, formatDuration, isActiveJob, jobModelFile, plannedEpochs,
} from './training';

const INFO = { min_audio_seconds: 60, recommended_audio_seconds: [300, 1800], max_audio_seconds: 3600 };

describe('training helpers', () => {
  it('formats durations compactly', () => {
    expect(formatDuration(12)).toBe('12 s');
    expect(formatDuration(250)).toBe('4 min 10 s');
    expect(formatDuration(3900)).toBe('1 h 05 min');
  });

  it('grades dataset length: five minutes is good', () => {
    expect(datasetQuality(0, INFO).tone).toBe('muted');
    expect(datasetQuality(30, INFO).tone).toBe('danger');
    expect(datasetQuality(120, INFO).tone).toBe('warn');
    expect(datasetQuality(300, INFO).tone).toBe('ok');
    expect(datasetQuality(900, INFO).tone).toBe('ok');
    expect(datasetQuality(4000, INFO).tone).toBe('danger');
  });

  it('knows which jobs run, resume and what they export', () => {
    expect(isActiveJob({ state: 'running' })).toBe(true);
    expect(canResume({ state: 'interrupted' })).toBe(true);
    expect(canResume({ state: 'running' })).toBe(false);
    expect(jobModelFile({ name: 'voz' })).toBe('voz.pth');
  });

  it('describes the kept epoch and the planned budget', () => {
    expect(bestEpochSummary({})).toBeNull();
    expect(bestEpochSummary({ exported_epoch: 140, best_score: 1.234 })).toContain('epoch 140');
    expect(bestEpochSummary({ exported_epoch: 25 })).toBe('Saved at epoch 25');
    expect(plannedEpochs({ plan: { epochs: 240 } })).toBe(240);
    expect(plannedEpochs({ epochs: null })).toBeNull();
  });
});
