import React, { memo } from 'react';
import { FALLBACK_F0_METHODS, findF0Method, groupedMethods } from '../lib/f0Methods';
import Button from './Button';
import Toggle from './Toggle';
import { BigSlider, MiniSlider } from './Sliders';

function pitchHint(voice, autoPitchAvailable, autoPitchShift, isRunning) {
  if (!voice.autoPitch) return 'Manual shift. Turn Auto on to match the voice register.';
  if (!isRunning) return 'Auto: matches your register to the voice once you speak.';
  if (!autoPitchAvailable) return 'Auto needs a voice trained in this studio; this one keeps your pitch.';
  if (autoPitchShift === null) return 'Auto: listening to your voice…';
  return 'Auto: your register is matched to the voice.';
}

function calibrationText(isRunning, speakerProfile) {
  if (!isRunning) return 'Calibrates to you as you speak.';
  if (!speakerProfile) return 'Learning your voice: keep talking…';
  if (speakerProfile.calibration < 1) return `Learning your voice… ${Math.round(speakerProfile.calibration * 100)}%`;
  return `Calibrated: ${Math.round(speakerProfile.median_hz)} Hz, range ±${speakerProfile.range_semitones} st`;
}

/** The Voice card: pitch, melody, timbre and the pitch tracker. */
function VoiceLab({
  voice, onChange, hasModel, isRunning, autoPitchAvailable = false, autoPitchShift = null,
  f0Pending = null, tooSlow = false, speakerProfile = null, expressivenessApplied = null, onRecalibrate,
  f0Methods = FALLBACK_F0_METHODS,
}) {
  const autoActive = voice.autoPitch && autoPitchAvailable && autoPitchShift !== null;
  const shownPitch = autoActive ? autoPitchShift : voice.pitch;
  const selectedMethod = findF0Method(f0Methods, voice.f0Method);
  const liveGroups = groupedMethods(f0Methods, 'realtime');
  const trackerState = tooSlow ? 'Too slow here'
    : !selectedMethod.available ? selectedMethod.reason
      : f0Pending === selectedMethod.id ? 'Loading…' : 'Ready';

  return (
    <section className="panel flex h-full min-h-0 flex-col p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="panel-kicker">Voice</p>
          <h2 className="panel-title truncate">Pitch & timbre</h2>
        </div>
        <span className={`shrink-0 rounded-sm border px-2 py-1 text-[11px] font-medium ${
          isRunning ? 'border-ok-line bg-ok-bg text-ok-fg' : 'border-line-strong bg-control text-fg-subtle'}`}>
          {isRunning ? 'Live' : 'Applies on start'}
        </span>
      </div>

      <div className="mt-4 min-h-0 flex-1 space-y-5 overflow-y-auto pr-1">
        <BigSlider
          label="Pitch"
          value={shownPitch}
          min={-12}
          max={12}
          step={0.5}
          unit=" st"
          onChange={(value) => onChange({ pitch: value })}
          onReset={() => onChange({ pitch: 0, autoPitch: true })}
          resetDisabled={voice.autoPitch && voice.pitch === 0}
          accessory={(
            <span className="inline-flex shrink-0 items-center gap-2 text-[11px] font-medium text-fg-subtle">
              <Toggle checked={voice.autoPitch} onChange={(autoPitch) => onChange({ autoPitch })} label="Automatic pitch" />
              Auto
            </span>
          )}
          hint={pitchHint(voice, autoPitchAvailable, autoPitchShift, isRunning)}
        />

        <div>
          <MiniSlider
            label="Expressiveness"
            value={voice.expressiveness}
            min={0.5}
            max={2}
            step={0.05}
            onChange={(value) => onChange({ expressiveness: value })}
            format={(v) => `×${v.toFixed(2)}`}
            hint={voice.autoPitch && autoPitchAvailable && expressivenessApplied != null
              ? `Auto: ×${expressivenessApplied.toFixed(2)}, the voice's own melodic range.`
              : 'Above 1 widens your rises and falls in pitch: a livelier voice.'}
          />
          <div className="mt-2 flex h-8 items-center justify-between gap-3">
            <span className="min-w-0 truncate text-[11px] text-fg-subtle" title={calibrationText(isRunning, speakerProfile)}>
              {calibrationText(isRunning, speakerProfile)}
            </span>
            <Button onClick={onRecalibrate} disabled={!isRunning || !onRecalibrate} className="w-28">Recalibrate</Button>
          </div>
        </div>

        <BigSlider
          label="Formant"
          value={voice.formant}
          min={-12}
          max={12}
          step={0.5}
          unit=" st"
          onChange={(value) => onChange({ formant: value })}
          onReset={() => onChange({ formant: 0 })}
        />

        <div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="voice-f0-method" className="min-w-0 truncate text-xs font-medium text-fg-muted">
              Pitch tracker{hasModel ? '' : ' · needs a voice'}
            </label>
            <span className={`shrink-0 text-[11px] font-medium ${tooSlow || !selectedMethod.available ? 'text-warn-fg' : 'text-ok-fg'}`}>
              {trackerState}
            </span>
          </div>
          <select id="voice-f0-method" value={selectedMethod.id} onChange={(event) => onChange({ f0Method: event.target.value })}
            className="native-select-safe field mt-2">
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
          <p className={`mt-1.5 text-[11px] leading-4 ${tooSlow ? 'text-warn-fg' : 'text-fg-subtle'}`}>
            {tooSlow ? 'Too slow for live audio on this device: the voice will stutter. RMVPE or a larger buffer keeps up.'
              : selectedMethod.description}
          </p>
        </div>
      </div>
    </section>
  );
}

export default memo(VoiceLab);
