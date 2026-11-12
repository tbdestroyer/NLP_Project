// Live mode audio: capture a microphone or a shared tab, cut it into utterances with a simple
// voice activity detector, and hand each utterance (16 kHz mono float32, what Whisper reads) to
// the page, which sends it to POST /api/live/transcribe.
//
// Nothing is recorded or stored: each utterance exists only until it has been transcribed.

export const SAMPLE_RATE = 16000;
const FRAME = 480; // 30 ms at 16 kHz
const FRAME_MS = (FRAME / SAMPLE_RATE) * 1000;

export const VAD = {
  minLevel: 0.006, // RMS below this is never speech
  noiseRatio: 3, // speech is this many times louder than the background noise
  startFrames: 3, // 90 ms of speech starts an utterance
  prerollFrames: 10, // keep 300 ms before the start, so the first word is not clipped
  endSilenceMs: 750, // this much silence ends an utterance
  maxUtteranceMs: 20000, // long monologues are cut here (Whisper reads 30 s at most)
  minVoicedMs: 300, // shorter blips (a cough, a click) are dropped
};

const TAP = `
class InclinationTap extends AudioWorkletProcessor {
  process(inputs) {
    const channels = inputs[0];
    if (channels && channels.length) {
      const out = new Float32Array(channels[0].length);
      for (const channel of channels) for (let i = 0; i < out.length; i += 1) out[i] += channel[i] / channels.length;
      this.port.postMessage(out, [out.buffer]);
    }
    return true;
  }
}
registerProcessor('inclination-tap', InclinationTap);
`;

let tapUrl = null;
async function addTap(context) {
  tapUrl ??= URL.createObjectURL(new Blob([TAP], { type: 'application/javascript' }));
  await context.audioWorklet.addModule(tapUrl);
}

/** Box-filter downsampler from the device rate to 16 kHz (enough for speech recognition). */
class Resampler {
  constructor(fromRate) {
    this.step = fromRate / SAMPLE_RATE;
    this.position = 0; // where the next output sample starts, in input samples
    this.carry = new Float32Array(0);
  }

  push(input) {
    const data = new Float32Array(this.carry.length + input.length);
    data.set(this.carry);
    data.set(input, this.carry.length);
    const out = [];
    let start = this.position;
    while (start + this.step <= data.length) {
      const from = Math.floor(start);
      const to = Math.max(from + 1, Math.floor(start + this.step));
      let sum = 0;
      for (let i = from; i < to; i += 1) sum += data[i];
      out.push(sum / (to - from));
      start += this.step;
    }
    const keep = Math.floor(start);
    this.carry = data.slice(keep);
    this.position = start - keep;
    return Float32Array.from(out);
  }
}

export async function openMicrophone() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot use the microphone. Use Chrome or Edge on http://localhost.');
  return navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
}

/** The customer's side: a shared browser tab (Meet, Teams, Zoom web) or the whole screen with
 * system audio. The video track is not needed and is stopped right away. */
export async function openCallAudio() {
  if (!navigator.mediaDevices?.getDisplayMedia) throw new Error('This browser cannot share tab audio. Use Chrome or Edge.');
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    systemAudio: 'include',
    selfBrowserSurface: 'exclude',
  });
  if (!stream.getAudioTracks().length) {
    stream.getTracks().forEach((track) => track.stop());
    throw new Error('No audio was shared. Share the call\'s tab and tick "Also share tab audio" (or share your entire screen with system audio).');
  }
  stream.getVideoTracks().forEach((track) => track.stop());
  return new MediaStream(stream.getAudioTracks());
}

/**
 * One audio source cut into utterances.
 *   onStart() when someone starts talking; what it returns comes back as the utterance's `tag`
 *     (the page uses it for who was speaking at the start).
 *   onUtterance({ audio, startedAt, endedAt, tag }) when they stop (times in ms, performance.now()).
 *   onLevel(rms, speaking) about 33 times a second, for the level meter.
 */
export class SpeechCapture {
  constructor(stream, { onStart, onUtterance, onLevel, onEnded }) {
    this.stream = stream;
    this.onStart = onStart;
    this.onUtterance = onUtterance;
    this.onLevel = onLevel;
    this.paused = false;
    this.noise = 0.01;
    this.pending = new Float32Array(0);
    this.preroll = [];
    this.frames = null; // frames of the current utterance, or null between utterances
    this.voiced = 0;
    this.startRun = 0;
    this.silenceMs = 0;
    this.startedAt = 0;
    stream.getAudioTracks().forEach((track) => track.addEventListener('ended', () => onEnded?.()));
  }

  async start() {
    this.context = new AudioContext();
    await addTap(this.context);
    this.resampler = new Resampler(this.context.sampleRate);
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, 'inclination-tap');
    this.node.port.onmessage = (event) => this.feed(this.resampler.push(event.data));
    // A silent gain keeps the graph running without playing the audio back.
    this.mute = this.context.createGain();
    this.mute.gain.value = 0;
    this.source.connect(this.node).connect(this.mute).connect(this.context.destination);
    if (this.context.state === 'suspended') await this.context.resume();
  }

  setPaused(paused) {
    this.paused = paused;
    if (paused) this.finish();
  }

  stop() {
    this.finish();
    this.node?.port.close();
    this.source?.disconnect();
    this.context?.close();
    this.stream.getTracks().forEach((track) => track.stop());
  }

  feed(samples) {
    const data = new Float32Array(this.pending.length + samples.length);
    data.set(this.pending);
    data.set(samples, this.pending.length);
    let offset = 0;
    for (; offset + FRAME <= data.length; offset += FRAME) this.frame(data.subarray(offset, offset + FRAME).slice());
    this.pending = data.slice(offset);
  }

  frame(frame) {
    let energy = 0;
    for (let i = 0; i < frame.length; i += 1) energy += frame[i] * frame[i];
    const rms = Math.sqrt(energy / frame.length);
    const speech = !this.paused && rms > Math.max(VAD.minLevel, this.noise * VAD.noiseRatio);
    // The noise floor follows quiet stretches quickly and loud ones slowly.
    if (!speech) this.noise = rms < this.noise ? 0.9 * this.noise + 0.1 * rms : 0.995 * this.noise + 0.005 * rms;
    this.onLevel?.(rms, this.frames !== null);
    if (this.paused) return;

    if (this.frames === null) {
      this.preroll.push(frame);
      if (this.preroll.length > VAD.prerollFrames) this.preroll.shift();
      this.startRun = speech ? this.startRun + 1 : 0;
      if (this.startRun >= VAD.startFrames) {
        this.frames = [...this.preroll];
        this.voiced = this.startRun;
        this.silenceMs = 0;
        this.startedAt = performance.now() - this.frames.length * FRAME_MS;
        this.tag = this.onStart?.();
        this.preroll = [];
      }
      return;
    }
    this.frames.push(frame);
    if (speech) { this.voiced += 1; this.silenceMs = 0; } else this.silenceMs += FRAME_MS;
    if (this.silenceMs >= VAD.endSilenceMs || this.frames.length * FRAME_MS >= VAD.maxUtteranceMs) this.finish();
  }

  finish() {
    const frames = this.frames;
    this.frames = null;
    this.startRun = 0;
    if (!frames || this.voiced * FRAME_MS < VAD.minVoicedMs) return;
    // Keep 200 ms of the trailing silence.
    const trailing = Math.max(0, Math.floor((this.silenceMs - 200) / FRAME_MS));
    const kept = frames.slice(0, frames.length - trailing);
    const audio = new Float32Array(kept.length * FRAME);
    kept.forEach((frame, index) => audio.set(frame, index * FRAME));
    this.onUtterance?.({ audio, startedAt: this.startedAt, endedAt: this.startedAt + kept.length * FRAME_MS, tag: this.tag });
  }
}

/** Word overlap of two transcripts (0..1). With speakers instead of headphones the microphone
 * also hears the customer; such a near-copy of a customer utterance is dropped. */
export function overlap(a, b) {
  const words = (text) => new Set(text.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean));
  const left = words(a);
  const right = words(b);
  if (!left.size || !right.size) return 0;
  let shared = 0;
  for (const word of left) if (right.has(word)) shared += 1;
  return shared / Math.min(left.size, right.size);
}
