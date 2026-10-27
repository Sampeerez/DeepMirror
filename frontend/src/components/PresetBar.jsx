import React, { memo, useCallback, useEffect, useState } from 'react';
import {
  AudioLines, Bot, Building2, Check, Megaphone, Mountain, Phone, Plus, Radio, RadioTower, Sparkles, Trash2, X,
} from 'lucide-react';
import Button from './Button';
import { fetchPresets, savePreset, deletePreset } from '../lib/api';

// One glyph per built-in sound; anything new falls back to the sparkle.
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

function PresetTile({ preset, active, onApply }) {
  const Icon = PRESET_ICONS[preset.id] || Sparkles;
  return (
    <button type="button" onClick={() => onApply(preset)} title={preset.description} aria-pressed={active}
      data-selected={active}
      className={`frost-control flex h-14 min-w-0 flex-col items-center justify-center gap-1 px-2 ${
        active ? 'text-fg' : 'text-fg-muted'}`}>
      <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
      <span className="max-w-full truncate text-xs font-medium">{preset.name}</span>
    </button>
  );
}

/**
 * The Studio's sound presets: the built-in ones as a row of equal tiles, the
 * user's own in a menu beside them, with saving in place.
 */
function PresetBar({ activePresetId, onApplyPreset, getCurrentSettings, refreshRevision = 0 }) {
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

  const activeUser = presets.user.find((preset) => preset.id === activePresetId) || null;

  const handleDelete = async () => {
    if (!activeUser) return;
    try {
      await deletePreset(activeUser.id);
      await load();
    } catch (err) {
      setError(err.message || 'Failed to delete preset');
    }
  };

  return (
    <section id="presets" aria-label="Presets" className="panel flex flex-col gap-3 p-3 lg:flex-row lg:items-center">
      <div className="grid min-w-0 flex-1 grid-cols-4 gap-2 sm:grid-cols-8">
        {presets.builtin.map((preset) => (
          <PresetTile key={preset.id} preset={preset} active={activePresetId === preset.id} onApply={onApplyPreset} />
        ))}
      </div>

      <div className="flex shrink-0 items-center gap-2 lg:w-80">
        {showSave ? (
          <>
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
              className="field min-w-0 flex-1"
            />
            <Button size="md" icon={Check} variant="primary" onClick={handleSave} loading={saving}
              disabled={!saveName.trim()} aria-label="Save preset" title="Save preset" />
            <Button size="md" icon={X} onClick={() => setShowSave(false)} aria-label="Cancel preset save" title="Cancel" />
          </>
        ) : error ? (
          <p className="min-w-0 flex-1 truncate text-xs text-danger-fg" role="alert" title={error}>{error}</p>
        ) : (
          <>
            <select aria-label="My presets" value={activeUser?.id || ''} disabled={presets.user.length === 0}
              onChange={(event) => {
                const preset = presets.user.find((item) => item.id === event.target.value);
                if (preset) onApplyPreset(preset);
              }}
              className="native-select-safe field min-w-0 flex-1">
              <option value="" disabled>{presets.user.length ? 'My presets' : 'No presets of mine yet'}</option>
              {presets.user.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
            </select>
            {activeUser ? (
              <Button size="md" icon={Trash2} variant="danger" onClick={handleDelete}
                aria-label={`Delete ${activeUser.name} preset`} title="Delete this preset" />
            ) : null}
            <Button size="md" icon={Plus} onClick={() => setShowSave(true)} title="Save the current sound as a preset">
              Save
            </Button>
          </>
        )}
      </div>
    </section>
  );
}

export default memo(PresetBar);
