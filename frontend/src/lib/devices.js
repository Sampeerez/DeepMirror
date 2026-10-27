// Audio device lists without repeats.
//
// Browsers list the same hardware more than once: Chrome adds "default" and
// "communications" entries that point at a real device already listed (the
// studio has its own "Default" option for that), and Windows can expose one
// microphone through several endpoints with the same name.

const VIRTUAL_IDS = new Set(['default', 'communications']);
const ROLE_PREFIX = /^(default|communications|predeterminado|comunicaciones)\s*-\s*/i;

/** The device's own name, without the role prefix the browser adds. */
export function deviceName(label) {
  return String(label || '').replace(ROLE_PREFIX, '').trim();
}

/** One entry per physical device: same name (or, unnamed, same group) means the same device. */
export function uniqueDevices(devices) {
  const list = (devices || []).filter((device) => device?.deviceId);
  const real = list.filter((device) => !VIRTUAL_IDS.has(device.deviceId));
  const seen = new Set();
  const unique = [];
  for (const device of real.length ? real : list) {
    const name = deviceName(device.label).toLowerCase();
    const key = name ? `name:${name}` : `group:${device.groupId || device.deviceId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(device);
  }
  return unique;
}
