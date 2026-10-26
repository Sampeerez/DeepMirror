import React, { memo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { isRecommendedVoice } from '../lib/voice';
import Button from './Button';
import EffectsRack from './EffectsRack';
import { MiniSlider, SwitchRow } from './Sliders';
import { readStored, writeStored } from '../lib/storage';

const TABS = [
  { id: 'cleanup', label: 'Clean-up' },
  { id: 'effects', label: 'Effects' },
  { id: 'advanced', label: 'Advanced' },
];
const TAB_KEY = 'ovc_sound_tab_v1';

// Only said when it is not the normal case.
function noiseHint(voice, denoiser) {
  if (voice.noiseReductionDb > 0 && denoiser === 'spectral') return 'Light filter: DeepFilterNet is not ready on this device.';
  return null;
}

function CleanupTab({ voice, onChange, denoiser }) {
  return (
    <div className="space-y-5">
      <SwitchRow label="Convert speech only" checked={voice.isolateVoice}
        onChange={(isolateVoice) => onChange({ isolateVoice })} />
      <MiniSlider
        label="Noise removal"
        value={voice.noiseReductionDb}
        min={0}
        max={60}
        step={1}
        onChange={(value) => onChange({ noiseReductionDb: value })}
        format={(v) => (v <= 0 ? 'Off' : v >= 60 ? 'Max' : `${Math.round(v)} dB`)}
        hint={noiseHint(voice, denoiser)}
      />
      <SwitchRow label="Match my volume" checked={voice.matchLoudness}
        onChange={(matchLoudness) => onChange({ matchLoudness })} />
      <MiniSlider
        label="Room ambience"
        value={voice.ambientDb}
        min={-60}
        max={0}
        step={1}
        disabled={!voice.isolateVoice}
        onChange={(value) => onChange({ ambientDb: value })}
        format={(v) => (v <= -60 ? 'Off' : `${Math.round(v)} dB`)}
      />
    </div>
  );
}

function AdvancedTab({ voice, onChange, onRestoreRecommended }) {
  const recommended = isRecommendedVoice(voice);
  return (
    <div className="space-y-4">
      <MiniSlider label="Index rate" value={voice.indexRate} min={0} max={1} step={0.05}
        onChange={(value) => onChange({ indexRate: value })} />
      <MiniSlider label="Protect" value={voice.protect} min={0} max={0.5} step={0.01}
        onChange={(value) => onChange({ protect: value })} />
      <MiniSlider label="RMS mix" value={voice.rmsMixRate} min={0} max={1} step={0.05}
        onChange={(value) => onChange({ rmsMixRate: value })} />
      <MiniSlider label="Filter radius" value={voice.filterRadius} min={0} max={7} step={1}
        onChange={(value) => onChange({ filterRadius: value })}
        format={(v) => (v >= 3 ? String(Math.round(v)) : 'Off')} />
      {voice.f0Method?.startsWith('mangio-crepe') ? (
        <MiniSlider label="Crepe hop length" value={voice.crepeHopLength} min={64} max={512} step={16}
          onChange={(value) => onChange({ crepeHopLength: value })} format={(value) => `${Math.round(value)} samples`} />
      ) : null}
      <Button icon={RotateCcw} onClick={onRestoreRecommended} disabled={recommended} className="w-full">
        {recommended ? 'Recommended settings' : 'Restore recommended'}
      </Button>
    </div>
  );
}

/** The Sound card: what happens around the model, in three fixed-height tabs. */
function SoundCard({ voice, onChange, onRestoreRecommended, effects, onEffectsChange, denoiser }) {
  const [tab, setTab] = useState(() => {
    const stored = readStored(TAB_KEY);
    return TABS.some(({ id }) => id === stored) ? stored : 'cleanup';
  });
  const listRef = useRef(null);
  const select = (id) => {
    setTab(id);
    writeStored(TAB_KEY, id);
  };
  const onKeyDown = (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const index = TABS.findIndex(({ id }) => id === tab);
    const next = TABS[(index + (event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length].id;
    select(next);
    listRef.current?.querySelector(`[data-tab="${next}"]`)?.focus();
  };

  return (
    <section className="panel flex h-full min-h-0 flex-col p-5">
      <div className="min-w-0">
        <p className="panel-kicker">Sound</p>
        <h2 className="panel-title truncate">Clean-up & effects</h2>
      </div>
      <div ref={listRef} role="tablist" aria-label="Sound" onKeyDown={onKeyDown}
        className="mt-4 grid grid-cols-3 gap-1 rounded-lg border border-[color:var(--frost-border)] bg-sunken p-1">
        {TABS.map(({ id, label }) => {
          const selected = tab === id;
          return (
            <button key={id} type="button" role="tab" id={`sound-tab-${id}`} aria-selected={selected}
              aria-controls="sound-panel" tabIndex={selected ? 0 : -1} data-tab={id} data-selected={selected}
              onClick={() => select(id)}
              className={`h-8 min-w-0 truncate rounded-md border px-2 text-xs font-medium ${selected
                ? 'frost-control text-fg'
                : 'border-transparent text-fg-muted hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]'}`}>
              {label}
            </button>
          );
        })}
      </div>
      <div id="sound-panel" role="tabpanel" aria-labelledby={`sound-tab-${tab}`}
        className="mt-4 min-h-0 flex-1 overflow-y-auto pr-1">
        {tab === 'cleanup' ? <CleanupTab voice={voice} onChange={onChange} denoiser={denoiser} /> : null}
        {tab === 'effects' ? (
          <EffectsRack effects={effects} formantShift={voice.formant} onEffectsChange={onEffectsChange} />
        ) : null}
        {tab === 'advanced' ? (
          <AdvancedTab voice={voice} onChange={onChange} onRestoreRecommended={onRestoreRecommended} />
        ) : null}
      </div>
    </section>
  );
}

export default memo(SoundCard);
