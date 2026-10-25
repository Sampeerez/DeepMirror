import { APP_SLUG } from './brand';
// Call mode: which audio devices are the virtual cable.

// The Linux device the studio creates is named after the app (older versions: OpenVoiceChanger).
const APP_DEVICE = new RegExp(`${APP_SLUG}|openvoicechanger`, 'i');
const CABLE_PATTERNS = [/cable input/i, /vb-audio virtual cable/i, /vb-cable/i, APP_DEVICE];
const CABLE_MIC_PATTERNS = [/cable output/i, /vb-cable/i, APP_DEVICE];

/** The output device the studio should play into during a call (the cable's input). */
export function findCableOutput(outputDevices = []) {
  return outputDevices.find((device) => /cable input/i.test(device.label || ''))
    || outputDevices.find((device) => CABLE_PATTERNS.some((pattern) => pattern.test(device.label || '')))
    || null;
}

/** True for a microphone that is the virtual cable itself (never capture from it: feedback). */
export function isCableMicrophone(device) {
  return CABLE_MIC_PATTERNS.some((pattern) => pattern.test(device?.label || ''));
}
