import { describe, expect, it } from 'vitest';
import { deviceName, uniqueDevices } from './devices';

const mic = (deviceId, label, groupId = 'g1') => ({ deviceId, label, groupId, kind: 'audioinput' });

describe('uniqueDevices', () => {
  it('drops the browser role entries and repeated names', () => {
    const listed = [
      mic('default', 'Default - Microphone (Blue Yeti)'),
      mic('communications', 'Communications - Microphone (Blue Yeti)'),
      mic('a1', 'Microphone (Blue Yeti)'),
      mic('a2', 'Microphone (Blue Yeti)'),
      mic('b1', 'Headset Microphone', 'g2'),
    ];
    expect(uniqueDevices(listed).map((device) => device.deviceId)).toEqual(['a1', 'b1']);
  });

  it('tells unnamed devices apart by group', () => {
    const listed = [mic('a', '', 'g1'), mic('b', '', 'g1'), mic('c', '', 'g2')];
    expect(uniqueDevices(listed).map((device) => device.deviceId)).toEqual(['a', 'c']);
  });

  it('keeps a lone default entry when nothing else is listed', () => {
    expect(uniqueDevices([mic('default', 'Default')]).map((device) => device.deviceId)).toEqual(['default']);
  });

  it('strips the role prefix from names', () => {
    expect(deviceName('Predeterminado - Altavoces (Realtek)')).toBe('Altavoces (Realtek)');
  });
});
