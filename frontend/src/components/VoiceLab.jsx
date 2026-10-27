import React, { memo } from 'react';
import { FALLBACK_F0_METHODS, findF0Method, groupedMethods } from '../lib/f0Methods';
import Button from './Button';
import Toggle from './Toggle';
import { BigSlider, MiniSlider } from './Sliders';

// Auto only works with voices trained here (they store their pitch); say so when it cannot.
function pitchHint(voice, autoPitchAvailable, isRunning) {
  if (voice.autoPitch && isRunning && !autoPitchAvailable) return 'Auto needs a voice trained in this studio.';
  return null;
}

function calibrationText(isRunning, speakerProfile) {
  if (speakerProfile?.from_profile) return `From My voice: ${Math.round(speakerProfile.median_hz)} Hz`;
  if (!isRunning) return 'Not calibrated yet';
  if (!speakerProfile || speakerProfile.calibration < 1) {
    return `Calibrating… ${Math.round((speakerProfile?.calibration || 0) * 100)}%`;
  }
  return `Your voice: ${Math.round(speakerProfile.median_hz)} Hz, ±${speakerProfile.range_semitones} st`;
}

/** The Voice card: pitch, melody, timbre and the pitch tracker. */
function VoiceLab({
  voice, onChange, hasModel, isRunning, autoPitchAvailable = false, autoPitchShift = null,
  f0Pending = null, tooSlow = false, speakerProfile = null, expressivenessApplied = null, onRecalibrate,
  f0Methods = FALLBACK_F0_METHODS,
}) {
  const autoActive = voice.autoPitch && autoPitchAvailable && autoPitchShift !== null;
  const autoExpressiveness = voice.autoPitch && autoPitchAvailable ? expressivenessApplied : null;
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
          hint={pitchHint(voice, autoPitchAvailable, isRunning)}
        />

        <div>
          <MiniSlider
            label="Expressiveness"
            value={voice.expressiveness}
            min={0.5}
            max={2}
            step={0.05}
            onChange={(value) => onChange({ expressiveness: value })}
            format={(v) => (autoExpressiveness != null ? `Auto ×${autoExpressiveness.toFixed(2)}` : `×${v.toFixed(2)}`)}
          />
          <div className="mt-2 flex h-8 items-center justify-between gap-3">
            <span className="min-w-0 truncate text-[11px] text-fg-subtle" title={calibrationText(isRunning, speakerProfile)}>
              {calibrationText(isRunning, speakerProfile)}
            </span>
            <Button onClick={onRecalibrate} disabled={!isRunning || !onRecalibrate} className="w-28">Recalibrate</Button>
          </div>
        </div>

        <MiniSlider
          label="Pitch correction"
          value={voice.pitchCorrection}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => onChange({ pitchCorrection: value })}
          format={(v) => (v <= 0 ? 'Off' : `${Math.round(v * 100)}%`)}
        />

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
                    {method.label}{method.available ? '' : ' (unavailable)'}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          {tooSlow ? (
            <p className="mt-1.5 text-[11px] leading-4 text-warn-fg">Too slow here: the voice will stutter. Use RMVPE.</p>
          ) : null}
        </div>
      </div>
    </section>
  );
}

export default memo(VoiceLab);
