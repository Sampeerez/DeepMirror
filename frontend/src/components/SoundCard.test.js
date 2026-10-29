// @vitest-environment jsdom
import React from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import SoundCard from './SoundCard';
import VoiceLab from './VoiceLab';
import { DEFAULT_VOICE } from '../lib/voice';
import { defaultEffects } from '../lib/effects';

const h = React.createElement;
afterEach(() => { cleanup(); localStorage.clear(); });

describe('Studio cards', () => {
  it('switches the Sound card between tabs inside the same box, without disclosures', () => {
    const { container } = render(h(SoundCard, {
      voice: DEFAULT_VOICE, onChange: () => {}, onRestoreRecommended: () => {},
      effects: defaultEffects(), onEffectsChange: () => {}, denoiser: 'neural',
    }));
    expect(container.querySelector('details')).toBeNull();
    expect(screen.getByText('Max')).toBeTruthy();
    expect(screen.queryByText(/Light filter/)).toBeNull();   // the normal case needs no note

    fireEvent.click(screen.getByRole('tab', { name: 'Effects rack' }));
    expect(screen.getByText('All off')).toBeTruthy();
    expect(screen.queryByText('Noise removal')).toBeNull();

    expect(screen.queryByRole('tab', { name: 'Advanced' })).toBeNull();   // in Voice › Settings now
    // The panel keeps the card's height: content scrolls inside it.
    expect(screen.getByRole('tabpanel').className).toContain('overflow-y-auto');
  });

  it('shows the whole Voice card at once', () => {
    const { container } = render(h(VoiceLab, {
      voice: DEFAULT_VOICE, onChange: () => {}, hasModel: true, isRunning: false,
    }));
    expect(container.querySelector('details')).toBeNull();
    for (const label of ['Pitch', 'Expressiveness', 'Formant']) expect(screen.getByLabelText(label)).toBeTruthy();
    expect(screen.queryByLabelText(/Pitch tracker/)).toBeNull();   // in Voice › Settings now
  });
});
