// "My voice": the lines read once to record the speaker's profile.
//
// The set covers what calibration needs to hear: the normal speaking register
// (statements), its top and bottom (questions, exclamations, a quiet line),
// held vowels, a rolled r, numbers and a short melody, about 90 seconds in all.

export const PROFILE_LANGUAGES = [
  { id: 'es', label: 'Español' },
  { id: 'en', label: 'English' },
];

export const PROFILE_TARGET_SECONDS = 90;

export const PROFILE_PROMPTS = {
  es: [
    { kind: 'Normal', text: 'Hola, esta es mi voz de todos los días, tranquila y sin prisa, como hablo normalmente.' },
    { kind: 'Normal', text: 'Ayer por la tarde fui al mercado, compré pan, fruta y un poco de queso para la cena.' },
    { kind: 'Pregunta', text: '¿De verdad crees que llegaremos a tiempo? ¿Y si mejor salimos un poco antes?' },
    { kind: 'Exclamación', text: '¡No me lo puedo creer! ¡Qué alegría verte por aquí después de tanto tiempo!' },
    { kind: 'Bajito', text: 'Ahora en voz baja, casi en secreto: no se lo cuentes a nadie, ¿vale?' },
    { kind: 'Más fuerte', text: 'Y ahora un poco más fuerte, como si llamaras a alguien desde lejos: ¡ven aquí, que empezamos!' },
    { kind: 'Vocales', text: 'Alarga cada vocal unos segundos: aaaa… eeee… iiii… oooo… uuuu…' },
    { kind: 'Erre', text: 'El perro de San Roque no tiene rabo porque Ramón Rodríguez se lo ha robado. Rrrrr.' },
    { kind: 'Números', text: 'Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve y diez.' },
    { kind: 'Melodía', text: 'Canturrea una melodía sencilla, por ejemplo «cumpleaños feliz», con «la la la».' },
    { kind: 'Normal', text: 'Para terminar, cuéntame en una frase qué tal ha ido tu día y qué vas a hacer luego.' },
  ],
  en: [
    { kind: 'Normal', text: 'Hi, this is my everyday voice, calm and unhurried, the way I usually talk.' },
    { kind: 'Normal', text: 'Yesterday afternoon I went to the market and bought bread, some fruit and a little cheese.' },
    { kind: 'Question', text: 'Do you really think we will make it in time? Should we leave a bit earlier?' },
    { kind: 'Exclamation', text: 'I can\'t believe it! It\'s so good to see you here after all this time!' },
    { kind: 'Quiet', text: 'Now quietly, almost a whisper: don\'t tell anyone, all right?' },
    { kind: 'Louder', text: 'And now a little louder, as if calling someone from across the room: come on, we\'re starting!' },
    { kind: 'Vowels', text: 'Hold each vowel for a couple of seconds: aaaa… eeee… iiii… oooo… uuuu…' },
    { kind: 'Rolled r', text: 'Roll an r if you can, then read: red lorry, yellow lorry, red lorry, yellow lorry. Rrrrr.' },
    { kind: 'Numbers', text: 'One, two, three, four, five, six, seven, eight, nine and ten.' },
    { kind: 'Melody', text: 'Hum or sing a simple tune, such as "happy birthday", on "la la la".' },
    { kind: 'Normal', text: 'To finish, tell me in one sentence how your day has been and what you will do next.' },
  ],
};

const REGISTER_LABELS = { deep: 'Deep', low: 'Low', middle: 'Middle', high: 'High' };

export function registerLabel(register) {
  return REGISTER_LABELS[register] || '—';
}

/** "+3.5 st" style label for a shift in semitones. */
export function formatShift(semitones) {
  if (typeof semitones !== 'number') return '—';
  const sign = semitones > 0 ? '+' : semitones < 0 ? '−' : '';
  return `${sign}${Math.abs(semitones).toFixed(semitones % 1 ? 1 : 0)} st`;
}

/** What the active voice's calibration says, in one line. */
export function calibrationSummary(voice, profile) {
  if (!voice) return 'Activate a voice to calibrate it to your profile.';
  if (!voice.uses_pitch) return `${voice.name} does not use pitch: nothing to calibrate.`;
  if (!voice.target) return `${voice.name} does not know its own register yet: give it a clip of the original voice.`;
  if (!profile) return `${voice.name} is ready; record your voice to calibrate it.`;
  const { pitch_shift: shift, expressiveness } = voice.calibration;
  return `${voice.name}: ${formatShift(shift)} and ${expressiveness}× melody, from your profile.`;
}
