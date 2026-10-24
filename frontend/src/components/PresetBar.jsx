import React, { memo, useCallback, useEffect, useState } from 'react';
import { Check, Plus, Trash2, X } from 'lucide-react';
import Button from './Button';
import { fetchPresets, savePreset, deletePreset } from '../lib/api';

function PresetChip({ preset, active, onApply, onDelete }) {
  return (
    <div className="group relative shrink-0">
      <Button size="md" onClick={() => onApply(preset)} title={preset.description} aria-pressed={active}
        selected={active} className={`max-w-48 ${active ? 'text-fg' : ''}`}>
        {preset.emoji ? <span aria-hidden="true" className="mr-1.5">{preset.emoji}</span> : null}
        {preset.name}
      </Button>
      {onDelete && (
        /* reveal-on-hover: tucked away until hover/focus on a mouse-driven
           device, always visible where there is no hover to reveal it. */
        <button
          onClick={(event) => {
            event.stopPropagation();
            onDelete(preset);
          }}
          title="Delete preset"
          aria-label={`Delete ${preset.name} preset`}
          className="reveal-on-hover absolute -right-2 -top-2 inline-flex h-6 w-6 items-center justify-center rounded-md border border-danger-line-strong bg-panel text-danger-fg shadow-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--frost-focus)]"
        >
          <Trash2 className="h-3 w-3" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

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

  const handleDelete = async (preset) => {
    try {
      await deletePreset(preset.id);
      await load();
    } catch (err) {
      setError(err.message || 'Failed to delete preset');
    }
  };

  // One row of fixed height: the chips scroll sideways, saving happens in place.
  return (
    <section id="presets" className="panel flex flex-wrap items-center gap-3 p-3 sm:flex-nowrap">
      <p className="shrink-0 text-xs font-semibold text-fg-secondary">Presets</p>

      <div className="order-last flex min-w-0 basis-full gap-2 overflow-x-auto py-1 sm:order-none sm:basis-auto sm:flex-1">
        {presets.builtin.map((preset) => (
          <PresetChip key={preset.id} preset={preset} active={activePresetId === preset.id} onApply={onApplyPreset} />
        ))}
        {presets.user.length > 0 && <div className="mx-1 w-px shrink-0 self-stretch bg-line-strong" />}
        {presets.user.map((preset) => (
          <PresetChip key={preset.id} preset={preset} active={activePresetId === preset.id} onApply={onApplyPreset}
            onDelete={handleDelete} />
        ))}
        {error ? <p className="self-center truncate text-xs text-danger-fg" role="alert" title={error}>{error}</p> : null}
      </div>

      <div className="ml-auto flex shrink-0 items-center gap-2">
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
              className="field w-40"
            />
            <Button size="md" icon={Check} variant="primary" onClick={handleSave} loading={saving}
              disabled={!saveName.trim()} aria-label="Save preset" title="Save preset" />
            <Button size="md" icon={X} onClick={() => setShowSave(false)} aria-label="Cancel preset save" title="Cancel" />
          </>
        ) : (
          <Button size="md" icon={Plus} onClick={() => setShowSave(true)} className="w-40">Save current</Button>
        )}
      </div>
    </section>
  );
}

export default memo(PresetBar);
