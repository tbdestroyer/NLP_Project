// Voices for the Live page's demo call: the browser's own speech synthesis (Web Speech API), so
// nothing is installed or sent anywhere. Chrome and Edge on Windows ship several English voices;
// Edge also has natural-sounding "Online" ones, which are preferred when present.
import { useEffect, useMemo, useState } from 'react';

const synth = () => (typeof window !== 'undefined' ? window.speechSynthesis : undefined);

// Names tried in order for each side. \bmale\b does not match "Female".
const AGENT_NAMES = [/\bguy\b/i, /\bandrew\b/i, /\bbrian\b/i, /\bchristopher\b/i, /\bdavis\b/i, /\beric\b/i, /\bryan\b/i, /\bdavid\b/i, /\bmark\b/i, /\bdaniel\b/i, /uk english male/i, /\bmale\b/i];
const CUSTOMER_NAMES = [/\baria\b/i, /\bjenny\b/i, /\bemma\b/i, /\bava\b/i, /\bmichelle\b/i, /\bsonia\b/i, /\bzira\b/i, /\bhazel\b/i, /\bsamantha\b/i, /\bsusan\b/i, /google us english/i, /\bfemale\b/i];
const quality = (voice) => (/natural|online|neural/i.test(voice.name) ? 0 : 1) + (/^en-us/i.test(voice.lang) ? 0 : 0.5);

export function pickVoices(list) {
  const english = list.filter((voice) => /^en([-_]|$)/i.test(voice.lang));
  if (!english.length) return { agent: null, customer: null, sameVoice: false };
  const best = (patterns, exclude) => {
    for (const pattern of patterns) {
      const hits = english.filter((voice) => voice !== exclude && pattern.test(voice.name)).sort((a, b) => quality(a) - quality(b));
      if (hits.length) return hits[0];
    }
    return [...english].sort((a, b) => quality(a) - quality(b)).find((voice) => voice !== exclude) ?? english[0];
  };
  const agent = best(AGENT_NAMES);
  const customer = best(CUSTOMER_NAMES, agent);
  return { agent, customer, sameVoice: agent === customer };
}

/** { supported, agent, customer, sameVoice }: voices load asynchronously in Chrome. */
export function useDemoVoices() {
  const [list, setList] = useState(() => synth()?.getVoices() ?? []);
  useEffect(() => {
    const engine = synth();
    if (!engine) return undefined;
    const update = () => setList(engine.getVoices());
    update();
    engine.addEventListener?.('voiceschanged', update);
    return () => engine.removeEventListener?.('voiceschanged', update);
  }, []);
  return useMemo(() => ({ supported: Boolean(synth()), ...pickVoices(list) }), [list]);
}

export const shortName = (voice) => voice?.name.replace(/^(Microsoft|Google)\s+/i, '').replace(/\s*\(.*$/, '').replace(/\s+-\s+.*$/, '').replace(/\s+Online.*$/i, '') ?? '';

/**
 * Speaks one line. onWord(spokenSoFar) as words are reached (not every voice reports words);
 * onDone() once, when the line ends, fails, or (if the browser never says so) after a fallback time.
 * Returns stop(), which silences the line without calling onDone.
 */
export function speakLine(text, { voice, pitch = 1, rate = 1.03, onWord, onDone }) {
  const engine = synth();
  let finished = false;
  let fallback = null;
  const done = () => {
    if (finished) return;
    finished = true;
    clearTimeout(fallback);
    onDone?.();
  };
  if (!engine) { setTimeout(done, 0); return () => { finished = true; }; }
  const utterance = new SpeechSynthesisUtterance(text);
  if (voice) { utterance.voice = voice; utterance.lang = voice.lang; }
  utterance.pitch = pitch;
  utterance.rate = rate;
  utterance.onboundary = (event) => {
    if (event.name && event.name !== 'word') return;
    const end = text.slice(event.charIndex).search(/\s|$/);
    onWord?.(text.slice(0, event.charIndex + Math.max(end, event.charLength ?? 0)));
  };
  utterance.onend = done;
  utterance.onerror = done;
  // Chrome sometimes never fires onend; about 13 characters a second is a slow speaking rate.
  fallback = setTimeout(done, 2500 + (text.length / 13) * 1000 / rate);
  // Chrome can drop an utterance that is garbage-collected mid-sentence; keep a reference.
  speakLine.current = utterance;
  engine.cancel();
  engine.speak(utterance);
  return () => {
    finished = true;
    clearTimeout(fallback);
    engine.cancel();
  };
}

export const stopSpeaking = () => synth()?.cancel();
