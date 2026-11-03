import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { describe, expect, it } from 'vitest';

function playback(chunkSize = 256) {
  const processors = {};
  const messages = [];
  const context = {
    AudioWorkletProcessor: class { port = { postMessage: (data) => messages.push(data) }; },
    registerProcessor: (name, processor) => { processors[name] = processor; },
    sampleRate: 48000,
    Float32Array,
  };
  vm.runInNewContext(readFileSync(new URL('../../public/audioWorklet.js', import.meta.url), 'utf8'), context);
  const processor = new processors['playback-processor']({ processorOptions: { chunkSize } });
  const tick = () => {
    const output = new Float32Array(128);
    processor.process([], [[output]]);
    return output;
  };
  return { processor, messages, tick, send: (data) => processor.port.onmessage({ data }) };
}

describe('playback buffering diagnostics', () => {
  it('does not count startup silence as underruns and reports at most twice a second', () => {
    const { processor, messages, tick } = playback();
    for (let i = 0; i < 375; i++) tick();
    expect(processor.underruns).toBe(0);
    expect(messages.length).toBeLessThanOrEqual(2);
    expect(messages[0].type).toBe('diagnostics');
  });

  it('counts one underrun per starvation episode, then recovers', () => {
    const { processor, tick, send } = playback();
    send(new Float32Array(256).fill(0.5));
    expect(tick()[0]).toBe(0.5);
    tick(); tick();
    for (let i = 0; i < 100; i++) tick();
    expect(processor.underruns).toBe(1);
    send(new Float32Array(256).fill(0.25));
    expect(tick()[0]).toBe(0.25);
    tick(); tick();
    expect(processor.underruns).toBe(2);
  });

  it('plays exactly what arrives while no cushion is asked for', () => {
    const { processor, tick, send } = playback(1024);
    send(Float32Array.from({ length: 1024 }, (_, i) => i));
    send(Float32Array.from({ length: 1024 }, (_, i) => 1024 + i));
    const heard = [...tick(), ...tick()];
    expect(heard).toEqual(Array.from({ length: 256 }, (_, i) => i));
    expect(processor.rate).toBe(1);
  });

  // Chunks arriving on time, one per chunk's length, ``count`` times; how much is left at the end.
  function play(player, count, chunkSize) {
    for (let c = 0; c < count; c++) {
      player.send(new Float32Array(chunkSize).fill(0.5));
      for (let t = 0; t < chunkSize / 128; t++) player.tick();
    }
    return player.processor.availableSamples;
  }

  it('gains the cushion it is asked for, with no gap', () => {
    const player = playback(1024);
    play(player, 2, 1024);
    player.send({ type: 'cushion', share: 0.5 });
    const left = play(player, 40, 1024);
    expect(left).toBeGreaterThanOrEqual(512 - 128);
    expect(player.processor.underruns).toBe(0);
    expect(player.processor.rate).toBe(1);
  });

  it('gives the cushion back once it is no longer asked for', () => {
    const player = playback(1024);
    play(player, 2, 1024);
    player.send({ type: 'cushion', share: 0.5 });
    play(player, 40, 1024);
    player.send({ type: 'cushion', share: 0 });
    const left = play(player, 60, 1024);
    expect(left).toBeLessThanOrEqual(1024 / 8 + 128);
    expect(player.processor.underruns).toBe(0);
    expect(player.processor.rate).toBe(1);
  });

  it('after a gap it keeps more in hand, up to one chunk', () => {
    const player = playback(1024);
    play(player, 2, 1024);
    for (let t = 0; t < 12; t++) player.tick();
    expect(player.processor.underruns).toBe(1);
    expect(player.processor.cushion).toBe(256);
    for (let g = 0; g < 8; g++) {
      player.send(new Float32Array(1024));
      for (let t = 0; t < 12; t++) player.tick();
    }
    expect(player.processor.cushion).toBeLessThanOrEqual(1024);
  });

  it('a stretched sound stays between its samples', () => {
    const player = playback(1024);
    player.send({ type: 'cushion', share: 1 });
    player.send(Float32Array.from({ length: 1024 }, (_, i) => i));
    const heard = player.tick();
    expect(heard[1]).toBeCloseTo(0.98, 5);
    for (let i = 1; i < 128; i++) expect(heard[i]).toBeGreaterThan(heard[i - 1]);
  });

  it('retains the newest audio across ring overflow and accounts for trimmed samples', () => {
    const { processor, tick, send } = playback();
    send(Float32Array.from({ length: 2000 }, (_, i) => i));
    expect(processor.availableSamples).toBe(512);
    expect(processor.droppedSamples).toBe(1488);
    expect(tick()[0]).toBe(1488);
  });
});
