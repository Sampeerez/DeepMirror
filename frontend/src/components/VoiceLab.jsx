import React, { memo, useState } from 'react';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { FALLBACK_F0_METHODS, findF0Method, groupedMethods } from '../lib/f0Methods';
import { isRecommendedVoice } from '../lib/voice';
import Toggle from './Toggle';

function BigSlider({ label, value, min, max, step, unit, onChange, onReset, resetDisabled, accessory, hint }) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <p className="text-xs font-medium text-fg-muted">{label}</p>
          {accessory}
        </div>
        <div className="flex items-center gap-2">
          <span className={`font-mono text-xl font-semibold tabular-nums tracking-normal ${value === 0 ? 'text-fg-subtle' : 'text-fg'}`}>
            {value > 0 ? '+' : ''}{Number(value).toFixed(1)}{unit}
          </span>
          <button
            onClick={onReset}
            disabled={resetDisabled ?? value === 0}
            className="chip-button inline-flex h-8 w-8 items-center justify-center !p-0"
            aria-label={`Reset ${label}`}
            title={`Reset ${label}`}
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-3 w-full"
      />
      <div className="mt-1.5 flex items-center justify-between text-[10px] font-medium text-fg-faint">
        <span>{min}</span>
        <span>0</span>
        <span>+{max}</span>
      </div>
      {hint ? <p className="mt-1.5 text-[11px] leading-4 text-fg-subtle">{hint}</p> : null}
    </div>
  );
}

function MiniSlider({ label, value, min, max, step, onChange, format = (v) => v.toFixed(2) }) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-fg-muted">{label}</span>
        <span className="font-mono text-xs tabular-nums text-fg-secondary">{format(Number(value))}</span>
      </div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="fx-slider mt-2 w-full"
      />
    </div>
  );
}

function pitchHint(voice, autoPitchAvailable, autoPitchShift, isRunning) {
  if (!voice.autoPitch) return 'Manual shift. Turn Auto on to match the voice register.';
  if (!isRunning) return 'Auto: matches your register to the voice once you speak (voices trained here).';
  if (!autoPitchAvailable) return 'Auto needs a voice trained in this studio; this voice keeps your pitch (0).';
  if (autoPitchShift === null) return 'Auto: listening to your voice…';
  return 'Auto: your register is matched to the voice. Move the slider to choose a shift yourself.';
}

function VoiceLab({
  voice, onChange, onRestoreRecommended, hasModel, isRunning, autoPitchAvailable = false, autoPitchShift = null,
  f0Pending = null, tooSlow = false, f0Methods = FALLBACK_F0_METHODS,
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const autoActive = voice.autoPitch && autoPitchAvailable && autoPitchShift !== null;
  const shownPitch = autoActive ? autoPitchShift : voice.pitch;
  const selectedMethod = findF0Method(f0Methods, voice.f0Method);
  const liveGroups = groupedMethods(f0Methods, 'realtime');

  return (
    <section className="panel p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="panel-kicker">Voice Lab</p>
          <h2 className="panel-title">Pitch & timbre</h2>
        </div>
        <span className={`rounded-sm border px-2 py-1 text-[11px] font-medium ${
          isRunning
            ? 'border-ok-line bg-ok-bg text-ok-fg'
            : 'border-line-strong bg-control text-fg-subtle'
        }`}
        >
          {isRunning ? 'Live' : 'Applies on start'}
        </span>
      </div>

      <div className="mt-5 space-y-6">
        <BigSlider
          label="Pitch Shift"
          value={shownPitch}
          min={-12}
          max={12}
          step={0.5}
          unit=" st"
          onChange={(value) => onChange({ pitch: value })}
          onReset={() => onChange({ pitch: 0, autoPitch: true })}
          resetDisabled={voice.autoPitch && voice.pitch === 0}
          accessory={(
            <span className="inline-flex items-center gap-2 text-[11px] font-medium text-fg-subtle">
              <Toggle
                checked={voice.autoPitch}
                onChange={(autoPitch) => onChange({ autoPitch })}
                label="Automatic pitch"
              />
              Auto
            </span>
          )}
          hint={pitchHint(voice, autoPitchAvailable, autoPitchShift, isRunning)}
        />

        <BigSlider
          label="Formant Shift"
          value={voice.formant}
          min={-12}
          max={12}
          step={0.5}
          unit=" st"
          onChange={(value) => onChange({ formant: value })}
          onReset={() => onChange({ formant: 0 })}
        />
      </div>

      <div className="mt-6 border-t border-line pt-5">
        <label htmlFor="voice-f0-method" className="block text-xs font-medium text-fg-muted">
          F0 Method {hasModel ? '' : '· needs an RVC model'}
        </label>
        <select
          id="voice-f0-method"
          value={selectedMethod.id}
          onChange={(event) => onChange({ f0Method: event.target.value })}
          className="native-select-safe mt-3 w-full rounded-lg border border-line-strong bg-input px-3 py-2.5 text-sm text-fg focus:border-line-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]"
        >
          {liveGroups.map((group) => (
            <optgroup key={group.section} label={group.section}>
              {group.methods.map((method) => (
                <option key={method.id} value={method.id} disabled={!method.available}>
                  {method.label}{method.available ? '' : ' — unavailable'}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <div className="mt-2 flex items-start justify-between gap-3">
          <p className="text-[11px] leading-4 text-fg-subtle">
            {selectedMethod.description}
            {tooSlow ? (
              <span className="mt-1 block text-warn-fg">
                Too slow for live audio on this device: the voice will stutter. RMVPE (or a larger buffer) keeps up.
              </span>
            ) : null}
          </p>
          <span className={`shrink-0 text-[11px] font-medium ${selectedMethod.available ? 'text-ok-fg' : 'text-warn-fg'}`}>
            {!selectedMethod.available ? selectedMethod.reason
              : f0Pending === selectedMethod.id ? 'Loading… (the previous one keeps working)' : 'Ready'}
          </span>
        </div>
      </div>

      <div className="mt-6 border-t border-line pt-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-medium text-fg-muted">Convert speech only</p>
            <p className="mt-1 text-[11px] leading-4 text-fg-subtle">
              Noise, breaths and clicks are not converted; the room stays audible as it is.
            </p>
          </div>
          <Toggle checked={voice.isolateVoice} onChange={(isolateVoice) => onChange({ isolateVoice })}
            label="Convert speech only" />
        </div>
        <div className={`mt-4 space-y-4 ${voice.isolateVoice ? '' : 'opacity-60'}`}>
          <MiniSlider
            label="Room ambience"
            value={voice.ambientDb}
            min={-60}
            max={0}
            step={1}
            onChange={(value) => onChange({ ambientDb: value })}
            format={(v) => (v <= -60 ? 'Off' : `${Math.round(v)} dB`)}
          />
        </div>
        <div className="mt-4">
          <MiniSlider
            label="Noise reduction (what the model hears)"
            value={voice.noiseReductionDb}
            min={0}
            max={24}
            step={1}
            onChange={(value) => onChange({ noiseReductionDb: value })}
            format={(v) => (v <= 0 ? 'Off' : `${Math.round(v)} dB`)}
          />
        </div>
        <div className="mt-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-medium text-fg-muted">Match my volume</p>
            <p className="mt-1 text-[11px] leading-4 text-fg-subtle">The converted voice is as loud as you speak.</p>
          </div>
          <Toggle checked={voice.matchLoudness} onChange={(matchLoudness) => onChange({ matchLoudness })}
            label="Match my volume" />
        </div>
      </div>

      <div className="mt-5 border-t border-line pt-4">
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          aria-expanded={showAdvanced}
          aria-controls="rvc-advanced-controls"
          className="frost-control flex min-h-9 w-full items-center justify-between px-3 py-2 text-xs font-medium text-fg-muted hover:text-fg"
        >
          RVC Advanced
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
            aria-hidden="true"
          />
        </button>

        {showAdvanced && (
          <div id="rvc-advanced-controls" className="mt-4 space-y-4">
            <MiniSlider
              label="Index Rate"
              value={voice.indexRate}
              min={0}
              max={1}
              step={0.05}
              onChange={(value) => onChange({ indexRate: value })}
            />
            <MiniSlider
              label="Filter Radius"
              value={voice.filterRadius}
              min={0}
              max={7}
              step={1}
              onChange={(value) => onChange({ filterRadius: value })}
              format={(v) => (v >= 3 ? String(Math.round(v)) : `${Math.round(v)} · off`)}
            />
            <MiniSlider
              label="RMS Mix"
              value={voice.rmsMixRate}
              min={0}
              max={1}
              step={0.05}
              onChange={(value) => onChange({ rmsMixRate: value })}
            />
            <MiniSlider
              label="Protect"
              value={voice.protect}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(value) => onChange({ protect: value })}
            />
            {voice.f0Method?.startsWith('mangio-crepe') ? (
              <MiniSlider
                label="Crepe Hop Length"
                value={voice.crepeHopLength}
                min={64}
                max={512}
                step={16}
                onChange={(value) => onChange({ crepeHopLength: value })}
                format={(value) => `${Math.round(value)} samples`}
              />
            ) : null}
            <p className="text-[11px] leading-4 text-fg-faint">
              Index rate blends retrieval features, RMS mix follows input loudness,
              protect preserves breaths and consonants. Filter radius median-smooths
              the Harvest and DIO pitch tracks — values below 3 disable it.
            </p>
            <button
              type="button"
              onClick={onRestoreRecommended}
              disabled={isRecommendedVoice(voice)}
              className="chip-button inline-flex items-center gap-2"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              {isRecommendedVoice(voice) ? 'Using recommended settings' : 'Restore recommended settings'}
            </button>
          </div>
        )}
      </div>
    </section>
  );
}

export default memo(VoiceLab);
