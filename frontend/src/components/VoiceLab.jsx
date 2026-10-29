import React, { memo } from 'react';
import { FALLBACK_F0_METHODS, findF0Method } from '../lib/f0Methods';
import Button from './Button';
import Toggle from './Toggle';
import { BigSlider, MiniSlider } from './Sliders';

// Auto only works with voices trained here (they store their pitch); say so when it cannot.
function pitchHint(voice, autoPitchAvailable, isRunning, hasModel) {
  if (voice.autoPitch && isRunning && hasModel && !autoPitchAvailable) return 'Auto needs a voice trained here';
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

/** Pitch, melody and timbre (the tracker and the model's detail are in Settings › Voice). */
function VoiceLab({
  voice, onChange, isRunning, autoPitchAvailable = false, autoPitchShift = null,
  f0Pending = null, tooSlow = false, speakerProfile = null, expressivenessApplied = null, onRecalibrate,
  f0Methods = FALLBACK_F0_METHODS, hasModel = true,
}) {
  const autoActive = voice.autoPitch && autoPitchAvailable && autoPitchShift !== null;
  const autoExpressiveness = voice.autoPitch && autoPitchAvailable ? expressivenessApplied : null;
  const shownPitch = autoActive ? autoPitchShift : voice.pitch;
  const selectedMethod = findF0Method(f0Methods, voice.f0Method);
  const trackerState = tooSlow ? 'Too slow here'
    : !selectedMethod.available ? selectedMethod.reason
      : f0Pending === selectedMethod.id ? 'Loading…' : 'Ready';

  return (
    <section className="panel flex min-h-0 flex-1 flex-col p-4">
      <h2 className="flex h-7 items-center text-sm font-semibold text-fg">Pitch & timbre</h2>

      <div className="mt-3 min-h-0 flex-1 space-y-3.5 overflow-y-auto pr-1">
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
          hint={pitchHint(voice, autoPitchAvailable, isRunning, hasModel)}
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

        {tooSlow || !selectedMethod.available || f0Pending === selectedMethod.id ? (
          <p className="text-[11px] leading-4 text-warn-fg" role="status">
            Pitch tracker {selectedMethod.label}: {trackerState}.
            {tooSlow ? ' Choose RMVPE in Settings.' : ''}
          </p>
        ) : null}
      </div>
    </section>
  );
}

export default memo(VoiceLab);
