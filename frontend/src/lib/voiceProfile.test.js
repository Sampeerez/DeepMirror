import { describe, expect, it } from 'vitest';
import {
  PROFILE_LANGUAGES, PROFILE_PROMPTS, calibrationSummary, formatShift, registerLabel,
} from './voiceProfile';

describe('voice profile prompts', () => {
  it('has the same varied set in every language', () => {
    const kinds = PROFILE_PROMPTS.es.length;
    for (const { id } of PROFILE_LANGUAGES) {
      expect(PROFILE_PROMPTS[id]).toHaveLength(kinds);
      expect(PROFILE_PROMPTS[id].every(({ text }) => text.length > 20)).toBe(true);
    }
  });
});

describe('calibration labels', () => {
  it('formats shifts with a sign', () => {
    expect(formatShift(12)).toBe('+12 st');
    expect(formatShift(-3.5)).toBe('−3.5 st');
    expect(formatShift(0)).toBe('0 st');
    expect(formatShift(null)).toBe('—');
    expect(registerLabel('deep')).toBe('Deep');
  });

  it('explains what the active voice still needs', () => {
    expect(calibrationSummary(null, null)).toMatch(/Activate a voice/);
    expect(calibrationSummary({ name: 'a.pth', uses_pitch: true, target: null }, {})).toMatch(/clip of the original/);
    expect(calibrationSummary({ name: 'a.pth', uses_pitch: true, target: {} }, null)).toMatch(/record your voice/);
    const voice = { name: 'a.pth', uses_pitch: true, target: {}, calibration: { pitch_shift: 9.5, expressiveness: 1.3 } };
    expect(calibrationSummary(voice, {})).toBe('a.pth: +9.5 st and 1.3× melody, from your profile.');
  });
});
