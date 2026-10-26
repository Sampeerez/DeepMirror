const definitions = [
  ['pm', 'PM', 'CPU, lowest load', 'Classic', true],
  ['harvest', 'Harvest', 'CPU, smooth', 'Classic', false],
  ['dio', 'DIO', 'CPU, light', 'Classic', true],
  ['crepe', 'Crepe Full', 'Neural', 'Neural', false],
  ['crepe-tiny', 'Crepe Tiny', 'Neural, small', 'Neural', false],
  ['mangio-crepe', 'Mangio-Crepe', 'Crepe, adjustable hop', 'Neural', false],
  ['mangio-crepe-tiny', 'Mangio-Crepe Tiny', 'Crepe Tiny, adjustable hop', 'Neural', false],
  ['rmvpe', 'RMVPE', 'Most accurate', 'Neural', true],
  ['fcpe', 'FCPE', 'Fast', 'Neural', true],
  ['hybrid[crepe+rmvpe]', 'Hybrid: Crepe + RMVPE', 'Median of two', 'Hybrid', false],
  ['hybrid[crepe+fcpe]', 'Hybrid: Crepe + FCPE', 'Median of two', 'Hybrid', false],
  ['hybrid[rmvpe+fcpe]', 'Hybrid: RMVPE + FCPE', 'Median of two', 'Hybrid', false],
  ['hybrid[crepe+rmvpe+fcpe]', 'Hybrid: Crepe + RMVPE + FCPE', 'Median of three', 'Hybrid', false],
];

export const FALLBACK_F0_METHODS = definitions.map(([id, label, description, section, realtime]) => ({
  id, label, description, section, realtime, offline: true,
  available: false, reason: 'Waiting for server capabilities', components: [],
}));

export const F0_METHOD_IDS = FALLBACK_F0_METHODS.map(({ id }) => id);

export function normalizeF0Capabilities(value) {
  const supplied = new Map(
    (Array.isArray(value) ? value : [])
      .filter((item) => item && F0_METHOD_IDS.includes(item.id))
      .map((item) => [item.id, item])
  );
  return FALLBACK_F0_METHODS.map((fallback) => {
    const item = supplied.get(fallback.id);
    if (!item) return fallback;
    return {
      ...fallback,
      available: item.available === true,
      reason: typeof item.reason === 'string' ? item.reason : null,
      realtime: item.realtime === true,
      offline: item.offline === true,
      components: Array.isArray(item.components) ? item.components.filter((part) => F0_METHOD_IDS.includes(part)) : [],
    };
  });
}

export function groupedMethods(methods, mode = 'realtime') {
  const result = [];
  for (const method of methods || FALLBACK_F0_METHODS) {
    if (!method[mode]) continue;
    let group = result.find(({ section }) => section === method.section);
    if (!group) {
      group = { section: method.section, methods: [] };
      result.push(group);
    }
    group.methods.push(method);
  }
  return result;
}

export function findF0Method(methods, id) {
  return (methods || FALLBACK_F0_METHODS).find((method) => method.id === id)
    || FALLBACK_F0_METHODS.find((method) => method.id === id)
    || FALLBACK_F0_METHODS[0];
}
