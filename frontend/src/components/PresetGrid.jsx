import React, { memo, useCallback, useEffect, useState } from 'react';
import {
  AudioLines, Bot, Building2, Check, Megaphone, Mountain, Phone, Plus, Radio, RadioTower, Sparkles, X,
} from 'lucide-react';
import Button from './Button';
import { fetchPresets, savePreset, deletePreset } from '../lib/api';

// One glyph per built-in sound; the user's own get the sparkle.
const PRESET_ICONS = {
  clean: AudioLines,
  radio: Radio,
  telephone: Phone,
  walkie: RadioTower,
  megaphone: Megaphone,
  robot: Bot,
  stadium: Building2,
  cave: Mountain,
};

const TILE = 'frost-control flex h-12 w-full min-w-0 flex-col items-center justify-center gap-0.5 px-1.5';

function PresetTile({ preset, active, onApply, onDelete = null }) {
  const Icon = PRESET_ICONS[preset.id] || Sparkles;
  return (
    <div className="group relative min-w-0">
      <button type="button" onClick={() => onApply(preset)} aria-pressed={active} data-selected={active}
        className={`${TILE} ${active ? 'text-fg' : 'text-fg-muted'}`}>
        <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
        <span className="max-w-full truncate text-[11px] font-medium">{preset.name}</span>
      </button>
      {onDelete ? (
        <button type="button" onClick={() => onDelete(preset)} aria-label={`Delete ${preset.name}`}
          className="absolute -right-1 -top-1 hidden h-5 w-5 items-center justify-center rounded-full border border-line-strong bg-overlay text-fg-muted hover:text-danger-fg focus-visible:flex group-hover:flex">
          <X className="h-3 w-3" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}

/**
 * The sound presets, inside the Effects card: the built-in ones and the
 * user's own as one grid of tiles. Pressing the active one again turns it off
 * (App keeps what was there before). The last tile saves the current sound.
 */
function PresetGrid({ activePresetId, onApplyPreset, getCurrentSettings, refreshRevision = 0 }) {
  const [presets, setPresets] = useState({ builtin: [], user: [] });
  const [saving, setSaving] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [showSave, setShowSave] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchPresets();
      setPresets({
        builtin: Array.isArray(data?.builtin) ? data.builtin : [],
        user: Array.isArray(data?.user) ? data.user : [],
      });
      setError(null);
    } catch {
      setError('Preset service unavailable');
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshRevision]);

  const handleSave = async () => {
    const name = saveName.trim();
    if (!name) return;
    setSaving(true);
    setError(null);
    try {
      await savePreset(name, getCurrentSettings());
      setSaveName('');
      setShowSave(false);
      await load();
    } catch (err) {
      setError(err.message || 'Failed to save preset');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (preset) => {
    try {
      await deletePreset(preset.id);
      await load();
    } catch (err) {
      setError(err.message || 'Failed to delete preset');
    }
  };

  return (
    <div id="presets" role="group" aria-label="Presets" className="shrink-0">
      <div className="grid max-h-[10.5rem] grid-cols-3 gap-1.5 overflow-y-auto p-0.5">
        {presets.builtin.map((preset) => (
          <PresetTile key={preset.id} preset={preset} active={activePresetId === preset.id} onApply={onApplyPreset} />
        ))}
        {presets.user.map((preset) => (
          <PresetTile key={preset.id} preset={preset} active={activePresetId === preset.id} onApply={onApplyPreset}
            onDelete={handleDelete} />
        ))}
        {!showSave ? (
          <button type="button" onClick={() => setShowSave(true)} aria-label="Save the current sound as a preset"
            className={`${TILE} border-dashed text-fg-subtle`}>
            <Plus className="h-4 w-4" aria-hidden="true" />
            <span className="text-[11px] font-medium">Save</span>
          </button>
        ) : null}
      </div>
      {showSave ? (
        <div className="mt-2 flex items-center gap-2">
          <input
            value={saveName}
            onChange={(event) => setSaveName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') handleSave();
              if (event.key === 'Escape') setShowSave(false);
            }}
            placeholder="Preset name"
            aria-label="Preset name"
            maxLength={40}
            autoFocus
            className="field h-9 min-w-0 flex-1"
          />
          <Button size="md" icon={Check} variant="primary" onClick={handleSave} loading={saving}
            disabled={!saveName.trim()} aria-label="Save preset" />
          <Button size="md" icon={X} onClick={() => setShowSave(false)} aria-label="Cancel" />
        </div>
      ) : null}
      {error ? <p className="mt-2 truncate text-xs text-danger-fg" role="alert" title={error}>{error}</p> : null}
    </div>
  );
}

export default memo(PresetGrid);
