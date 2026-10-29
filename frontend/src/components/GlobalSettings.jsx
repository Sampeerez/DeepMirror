import React, { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { Moon, ScrollText, Settings2, Sun, X } from 'lucide-react';
import Button from './Button';
import RuntimeReadiness from './RuntimeReadiness';
import Segmented from './Segmented';
import SettingRow, { SettingsSection } from './SettingRow';
import Toggle from './Toggle';
import { APP_NAME } from '../lib/brand';

const LogsPanel = lazy(() => import('./LogsPanel'));

const THEMES = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
];
const SECTIONS = [
  { value: 'general', label: 'General', icon: Settings2 },
  { value: 'logs', label: 'Logs', icon: ScrollText },
];

/**
 * Settings that belong to the whole studio, not to the voice or the face
 * (those are in Voice › Settings and Face › Settings): appearance, the engine,
 * test samples, and the server's log.
 */
export default function GlobalSettings({
  config, onClose, theme, onThemeChange, runtimeChecking, runtimeError, onRefreshRuntime,
  captureSamples = false, onCaptureSamplesChange, initialSection = 'general', version = null,
}) {
  const dialogRef = useRef(null);
  const [section, setSection] = useState(initialSection);

  useEffect(() => {
    const opener = document.activeElement;
    dialogRef.current?.querySelector('button')?.focus({ preventScroll: true });
    return () => {
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);

  // Focus stays inside the dialog.
  const handleDialogKeyDown = (event) => {
    if (event.key !== 'Tab') return;
    const controls = [...dialogRef.current.querySelectorAll('button, input, select, a[href], [tabindex]')]
      .filter((element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length > 0);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  };

  return (
    <section ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="settings-title"
      onKeyDown={handleDialogKeyDown}
      className="flex max-h-[calc(100svh-2.5rem)] flex-col rounded-lg border border-line-strong bg-overlay p-6 shadow-[0_16px_64px_var(--overlay-shadow)]">
      <div className="flex items-center justify-between gap-4 border-b border-line pb-4">
        <h2 id="settings-title" className="text-lg font-semibold text-fg">Settings</h2>
        <div className="flex items-center gap-3">
          <Segmented label="Section" hideLabel options={SECTIONS} value={section} onChange={setSection}
            className="w-56" />
          {onClose ? (
            <Button icon={X} size="md" variant="ghost" onClick={onClose} aria-label="Close settings" title="Close settings" />
          ) : null}
        </div>
      </div>

      {section === 'general' ? (
        <div className="mt-5 grid min-h-0 gap-4 overflow-y-auto lg:grid-cols-2">
          <div className="space-y-4">
            <SettingsSection title="Appearance">
              <SettingRow label="Theme">
                <Segmented label="Theme" hideLabel options={THEMES} value={theme} onChange={onThemeChange}
                  className="w-48" />
              </SettingRow>
            </SettingsSection>
            <SettingsSection title="Tests">
              <SettingRow label="Save my microphone with each take"
                hint="The raw and the converted audio go to data/samples, to find a problem.">
                <Toggle checked={captureSamples} onChange={onCaptureSamplesChange}
                  label="Save my microphone with each take" />
              </SettingRow>
            </SettingsSection>
            <p className="px-1 text-[11px] text-fg-faint">
              {APP_NAME}{version ? ` ${version}` : ''}. The voice and the face have their own settings in
              Voice › Settings and Face › Settings.
            </p>
          </div>
          <SettingsSection title="Engine">
            <div className="px-4 py-3">
              <RuntimeReadiness config={config} checking={runtimeChecking} error={runtimeError}
                onRefresh={onRefreshRuntime} />
            </div>
          </SettingsSection>
        </div>
      ) : (
        <div className="mt-5 flex min-h-[60svh] flex-col">
          <Suspense fallback={null}>
            <LogsPanel active />
          </Suspense>
        </div>
      )}
    </section>
  );
}
