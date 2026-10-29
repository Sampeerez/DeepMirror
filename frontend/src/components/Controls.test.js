// @vitest-environment jsdom
import React, { useState } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import Picker from './Picker';
import Segmented from './Segmented';

const h = React.createElement;
afterEach(cleanup);

function Choice({ initial = 'auto' }) {
  const [value, setValue] = useState(initial);
  return h(Segmented, {
    label: 'Mouth', value, onChange: setValue,
    options: [{ value: 'auto', label: 'Auto' }, { value: 'on', label: 'Always' }, { value: 'off', label: 'Never' }],
  });
}

describe('Segmented', () => {
  it('is a labelled radiogroup with one option in the Tab order', () => {
    render(h(Choice));
    expect(screen.getByRole('radiogroup', { name: 'Mouth' })).toBeTruthy();
    const radios = screen.getAllByRole('radio');
    expect(radios.map((radio) => radio.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
    expect(radios.filter((radio) => radio.tabIndex === 0)).toHaveLength(1);
  });

  it('moves and chooses with the arrow keys, and wraps around', () => {
    render(h(Choice));
    const group = screen.getByRole('radiogroup');
    fireEvent.keyDown(group, { key: 'ArrowRight' });
    expect(screen.getByRole('radio', { name: 'Always' }).getAttribute('aria-checked')).toBe('true');
    fireEvent.keyDown(group, { key: 'ArrowLeft' });
    fireEvent.keyDown(group, { key: 'ArrowLeft' });
    expect(screen.getByRole('radio', { name: 'Never' }).getAttribute('aria-checked')).toBe('true');
  });
});

function Devices() {
  const [value, setValue] = useState('');
  return h(Picker, {
    label: 'Microphone', value, onChange: setValue,
    options: [{ value: '', label: 'Default microphone' }, { value: 'a', label: 'Blue Yeti' }, { value: 'b', label: 'Headset' }],
  });
}

describe('Picker', () => {
  it('shows the choice on a button and opens a listbox only when asked', () => {
    render(h(Devices));
    const button = screen.getByRole('button', { name: 'Microphone: Default microphone' });
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByRole('listbox')).toBeNull();
    fireEvent.click(button);
    expect(screen.getByRole('listbox', { name: 'Microphone' })).toBeTruthy();
    expect(screen.getAllByRole('option')).toHaveLength(3);
  });

  it('chooses with the keyboard and closes, Escape closes without choosing', () => {
    render(h(Devices));
    const button = screen.getByRole('button', { name: /Microphone/ });
    fireEvent.keyDown(button, { key: 'ArrowDown' });
    fireEvent.keyDown(button, { key: 'ArrowDown' });
    fireEvent.keyDown(button, { key: 'Enter' });
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(screen.getByRole('button', { name: 'Microphone: Blue Yeti' })).toBeTruthy();
    fireEvent.keyDown(button, { key: 'ArrowDown' });
    fireEvent.keyDown(button, { key: 'Escape' });
    expect(screen.queryByRole('listbox')).toBeNull();
    expect(screen.getByRole('button', { name: 'Microphone: Blue Yeti' })).toBeTruthy();
  });
});

describe('Picker with long names', () => {
  it('shows every name whole in the list, so similar devices can be told apart', () => {
    const long = 'Micrófono (Realtek(R) Audio) — Front panel headset input, 48 kHz';
    render(h(Picker, {
      label: 'Microphone', value: '', onChange: () => {},
      options: [{ value: '', label: 'Default microphone' }, { value: 'a', label: long }],
    }));
    fireEvent.click(screen.getByRole('button', { name: /Microphone/ }));
    const option = screen.getByRole('option', { name: long });
    expect(option.textContent).toContain(long);
    expect(option.querySelector('.truncate')).toBeNull();
  });
});
