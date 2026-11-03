// AudioWorklet processor file — loaded via audioContext.audioWorklet.addModule()
// Contains CaptureProcessor and PlaybackProcessor.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunkSize = options.processorOptions?.chunkSize || 4096;
    this.buffer = new Float32Array(this.chunkSize);
    this.writeIndex = 0;
    // A longer chunk while running (more headroom when the computer is busy).
    this.port.onmessage = ({ data }) => {
      if (data?.type !== 'chunkSize' || !(data.chunkSize > 0)) return;
      const next = new Float32Array(data.chunkSize);
      next.set(this.buffer.subarray(0, Math.min(this.writeIndex, data.chunkSize)));
      this.buffer = next;
      this.chunkSize = data.chunkSize;
      this.writeIndex = Math.min(this.writeIndex, data.chunkSize - 1);
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.writeIndex++] = channelData[i];

      if (this.writeIndex >= this.chunkSize) {
        // Send a copy of the buffer to the main thread
        this.port.postMessage(this.buffer.slice(0));
        this.writeIndex = 0;
      }
    }

    return true;
  }
}

// A cushion is gained or given back by playing this much slower or faster: no silence, and a 2 %
// change for a few seconds does not show.
const CUSHION_RATE = 0.02;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkSize = options.processorOptions?.chunkSize || 4096;
    this.chunkSize = chunkSize;
    this.cushion = 0;           // samples to still have when the next chunk arrives
    this.draining = false;      // a cushion no longer asked for is being given back
    this.rate = 1;
    this.phase = 0;
    this.primeBufferSize = chunkSize;
    this.targetBufferSize = 2 * chunkSize;
    this.ringBufferSize = 6 * chunkSize;
    this.ringBuffer = new Float32Array(this.ringBufferSize);
    this.readIndex = 0;
    this.writeIndex = 0;
    this.availableSamples = 0;
    this.isPrimed = false;
    this.lastSample = 0;
    this.underruns = 0;
    this.droppedSamples = 0;
    this.reportSamples = 0;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data?.type === 'chunkSize' && data.chunkSize > 0) {
        this.resize(data.chunkSize);
        return;
      }
      if (data?.type === 'cushion') {
        this.cushion = Math.round(data.share * this.chunkSize);
        this.draining = this.cushion === 0;
        return;
      }
      if (data instanceof Float32Array) {
        this.adjustRate();
        for (let i = 0; i < data.length; i++) {
          if (this.availableSamples >= this.ringBufferSize) {
            this.readIndex = (this.readIndex + 1) % this.ringBufferSize;
            this.availableSamples -= 1;
            this.droppedSamples += 1;
          }
          this.ringBuffer[this.writeIndex] = data[i];
          this.writeIndex = (this.writeIndex + 1) % this.ringBufferSize;
          this.availableSamples += 1;
        }

        while (this.availableSamples > this.targetBufferSize + this.cushion) {
          this.readIndex = (this.readIndex + 1) % this.ringBufferSize;
          this.availableSamples -= 1;
          this.droppedSamples += 1;
        }

        if (!this.isPrimed && this.availableSamples >= this.primeBufferSize) {
          this.isPrimed = true;
        }
      }
    };
  }

  /** When a chunk arrives: what is left of the last one says whether the cushion is there. */
  adjustRate() {
    const left = this.availableSamples;
    if (left < this.cushion) {
      this.rate = 1 - CUSHION_RATE;
    } else if (this.draining && left > this.chunkSize / 8) {
      this.rate = 1 + CUSHION_RATE;
    } else {
      this.rate = 1;
      this.draining = false;
    }
  }

  /** Buffer targets for a new chunk size; queued audio is kept. */
  resize(chunkSize) {
    this.cushion = Math.round((this.cushion / this.chunkSize) * chunkSize);
    this.chunkSize = chunkSize;
    const ring = new Float32Array(6 * chunkSize);
    const keep = Math.min(this.availableSamples, ring.length);
    for (let i = 0; i < keep; i++) {
      ring[i] = this.ringBuffer[(this.readIndex + this.availableSamples - keep + i) % this.ringBufferSize];
    }
    this.ringBuffer = ring;
    this.ringBufferSize = ring.length;
    this.readIndex = 0;
    this.writeIndex = keep % ring.length;
    this.availableSamples = keep;
    this.primeBufferSize = chunkSize;
    this.targetBufferSize = 2 * chunkSize;
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channelData = output[0];

    if (!this.isPrimed && this.availableSamples < this.primeBufferSize) {
      for (let i = 0; i < channelData.length; i++) {
        channelData[i] = 0;
      }
    } else {
      for (let i = 0; i < channelData.length; i++) {
        if (this.availableSamples <= 0) {
          this.isPrimed = false;
          this.underruns += 1;
          // A gap: the next ones are avoided by keeping more in hand from now on.
          this.cushion = Math.min(this.cushion + Math.round(this.chunkSize / 4), this.chunkSize);
          let sample = this.lastSample;
          for (let j = i; j < channelData.length; j++) {
            sample *= 0.985;
            channelData[j] = sample;
          }
          this.lastSample = sample;
          break;
        }

        let sample = this.ringBuffer[this.readIndex];
        if (this.rate === 1 && this.phase === 0) {
          this.readIndex = (this.readIndex + 1) % this.ringBufferSize;
          this.availableSamples -= 1;
        } else {
          if (this.availableSamples > 1) {
            sample += this.phase * (this.ringBuffer[(this.readIndex + 1) % this.ringBufferSize] - sample);
          }
          this.phase += this.rate;
          const whole = Math.min(Math.floor(this.phase), this.availableSamples);
          this.phase = this.rate === 1 ? 0 : this.phase - whole;
          this.readIndex = (this.readIndex + whole) % this.ringBufferSize;
          this.availableSamples -= whole;
        }
        channelData[i] = sample;
        this.lastSample = sample;
      }
    }

    if (!this.isPrimed && this.availableSamples === 0) {
      this.lastSample = 0;
    }

    // Copy mono to all output channels
    for (let ch = 1; ch < output.length; ch++) {
      output[ch].set(channelData);
    }

    this.reportSamples += channelData.length;
    if (this.reportSamples >= sampleRate / 2) {
      this.reportSamples = 0;
      this.port.postMessage({ type: 'diagnostics', bufferedSamples: this.availableSamples,
        underruns: this.underruns, droppedSamples: this.droppedSamples });
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
registerProcessor('playback-processor', PlaybackProcessor);
