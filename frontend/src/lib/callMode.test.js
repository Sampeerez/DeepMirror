import { describe, expect, it } from 'vitest';
import { findCableOutput, isCableMicrophone } from './callMode';

describe('call mode devices', () => {
  const outputs = [
    { deviceId: 'a', label: 'Altavoces (Razer)' },
    { deviceId: 'b', label: 'CABLE Input (VB-Audio Virtual Cable)' },
  ];

  it('finds the virtual cable to play into', () => {
    expect(findCableOutput(outputs).deviceId).toBe('b');
    expect(findCableOutput([{ deviceId: 'l', label: 'OpenVoiceChanger_Output' }]).deviceId).toBe('l');
    expect(findCableOutput([outputs[0]])).toBeNull();
  });

  it('never captures from the cable itself', () => {
    expect(isCableMicrophone({ label: 'CABLE Output (VB-Audio Virtual Cable)' })).toBe(true);
    expect(isCableMicrophone({ label: 'Micrófono (CORSAIR VIRTUOSO)' })).toBe(false);
  });
});
