import { useEffect, useMemo, useRef, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import {
  ArrowDownRight, ArrowUpRight, Ban, Check, ChevronDown, Copy, Download, Keyboard, Lightbulb, LoaderCircle, Mic,
  MonitorSpeaker, Pause, PhoneOff, Play, Quote, Radio, Repeat2, RotateCcw, Send, ShieldCheck, SkipForward, Sparkles, Trash2, TriangleAlert,
  AudioLines, Volume2, VolumeX,
} from 'lucide-react';
import { getAnalyzeModels, getLiveStatus, getMetrics, loadSpeechModel, scoreLive, transcribe } from '../api/client';
import { openCallAudio, openMicrophone, overlap, SpeechCapture } from '../live/audio';
import { shortName, speakLine, stopSpeaking, useDemoVoices } from '../live/voices';
import { cap, pct, pts, Segmented, StatusTag } from '../components/ui';
import { CUSTOMER_CUES, DATASETS, MODELS, pretty } from '../data/taxonomy';
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/primitives';

const DOMAINS = { saas: 'Sales call or chat', craigslist: 'Price negotiation', p4g: 'Persuasion or donation ask' };

const MODES = [
  { value: 'call', icon: MonitorSpeaker, label: 'Microphone + call audio', audio: true,
    text: 'Your microphone is you. A shared browser tab (Google Meet, Teams or Zoom on the web), or your screen with system audio, is the customer. Wear headphones so your microphone does not pick up the customer.' },
  { value: 'mic', icon: Mic, label: 'Microphone only', audio: true,
    text: 'For a meeting in person or on speakerphone. Press Space (or the switch) whenever the other person starts talking.' },
  { value: 'text', icon: Keyboard, label: 'Type, or play a demo', audio: false,
    text: 'No audio. Type messages as either side, or play a demo call to watch the coach work.' },
];

// A SaaS discovery call with a price objection in the middle, for trying the page without a call.
const DEMO = [
  ['agent', "Hi Jordan, thanks for making time today. How's your week going?"],
  ['customer', "Busy, honestly. We're trying to sort out our support tooling."],
  ['agent', "Got it. What's the biggest headache with your current setup?"],
  ['customer', 'Tickets get lost between email and chat, and our response times are slipping.'],
  ['agent', 'Our platform puts email, chat and phone in one queue, so nothing slips through.'],
  ['customer', 'That sounds useful. How much is it for about 30 agents?'],
  ['agent', "It's $45 per agent per month, plus a one-time onboarding fee."],
  ['customer', "Hmm, that's more than we budgeted. I'm not sure we can justify it."],
  ['agent', 'I understand. Teams your size usually cut response times by a third in the first month, which pays for the seats.'],
  ['customer', "Okay, that's interesting. Do you have a case study I could share with my manager?"],
  ['agent', "Absolutely, I'll send you one from a similar company today."],
  ['customer', 'Great. What would the next steps be if we wanted to try it?'],
  ['agent', 'We can set up a two-week pilot this Friday. Shall I send the invite?'],
  ['customer', 'Sounds good, send it over.'],
];
const DEMO_PAUSE = { agent: 2600, customer: 4200 }; // ms after each silent line, so the coach can be read
const DEMO_GAP_SPOKEN = { agent: 700, customer: 2200 }; // ms after each spoken line (speaking already takes time)
const VOICES_KEY = 'inclination-demo-voices';

const ECHO_WINDOW_MS = 4000; // a microphone utterance this close to a near-identical call utterance is an echo
const EVIDENCE = {
  strong: { label: 'Strong evidence', className: 'border-teal-200 bg-teal-50 text-teal-800 dark:border-teal-900 dark:bg-teal-950/40 dark:text-teal-100' },
  suggestive: { label: 'Suggestive', className: 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-100' },
  typical: { label: 'About typical', className: 'border-border bg-muted text-muted-foreground' },
};

let nextId = 1;
const clock = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const speakerLabel = (role, names) => (role === 'agent' ? 'Sales rep' : names.customer);

/** Transcribed segments → conversation turns: echoes dropped, in time order, consecutive segments of one side merged. */
function toTurns(segments) {
  const done = segments.filter((s) => s.status === 'done' && s.text);
  const echo = new Set(done.filter((s) => s.source === 'mic' && done.some((o) => o.source === 'call'
    && Math.abs(o.start - s.start) < ECHO_WINDOW_MS && overlap(o.text, s.text) >= 0.6)).map((s) => s.id));
  const turns = [];
  for (const segment of [...done].sort((a, b) => a.start - b.start)) {
    if (echo.has(segment.id)) continue;
    const last = turns.at(-1);
    if (last && last.role === segment.role) { last.text += ` ${segment.text}`; last.ids.push(segment.id); } else turns.push({ role: segment.role, text: segment.text, ids: [segment.id] });
  }
  return { turns, echoes: echo.size };
}

export default function Live() {
  const [trained, setTrained] = useState(null);
  const [aucs, setAucs] = useState({});
  const [dataset, setDataset] = useState('saas');
  const [model, setModel] = useState('v1');
  const [mode, setMode] = useState('call');
  const [speech, setSpeech] = useState({ state: 'checking' }); // checking | loading | ready | missing | error
  const [phase, setPhase] = useState('setup'); // setup | live | ended
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [segments, setSegments] = useState([]);
  const [speaker, setSpeaker] = useState('agent'); // microphone-only mode: who is talking now
  const [paused, setPaused] = useState(false);
  const [levels, setLevels] = useState({ agent: 0, customer: 0 });
  const [startedAt, setStartedAt] = useState(null);
  const [endedAt, setEndedAt] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [result, setResult] = useState(null);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState('');
  const [demo, setDemo] = useState(null); // { step, playing, speaking } while the demo runs
  const voices = useDemoVoices();
  const [voicesOn, setVoicesOnState] = useState(() => { try { return localStorage.getItem(VOICES_KEY) !== 'off'; } catch { return true; } });
  const line = useRef(null); // { stop, finish } of the demo line being spoken

  const captures = useRef([]);
  const levelRef = useRef({ agent: 0, customer: 0 });
  const speakerRef = useRef(speaker);
  const pump = useRef({ busy: false, wanted: null });
  const selection = useRef({ dataset, model });
  speakerRef.current = speaker;
  selection.current = { dataset, model };

  const names = DATASETS[dataset] ?? { customer: 'Customer', agent: 'Sales rep' };
  const domains = Object.keys(DOMAINS).filter((key) => trained?.[key]?.length);
  const availableModels = (trained?.[dataset] ?? []).filter((key) => key in MODELS);
  const audioMode = MODES.find((m) => m.value === mode)?.audio;
  const { turns, echoes } = useMemo(() => toTurns(segments), [segments]);
  const pending = segments.filter((s) => s.status === 'transcribing');
  const turnsKey = JSON.stringify(turns.map((t) => [t.role, t.text]));

  // ---- setup ---------------------------------------------------------------------------------
  useEffect(() => { getAnalyzeModels().then(setTrained).catch((e) => { setError(e.message); setTrained({}); }); }, []);
  useEffect(() => { if (domains.length && !domains.includes(dataset)) setDataset(domains[0]); }, [domains.join(','), dataset]);
  useEffect(() => { if (availableModels.length && !availableModels.includes(model)) setModel(availableModels.includes('v1') ? 'v1' : availableModels[0]); }, [availableModels.join(','), model]);
  useEffect(() => {
    let alive = true;
    setAucs({});
    Promise.all(availableModels.map((key) => getMetrics({ dataset, model: key }).then((m) => [key, m?.roc_auc]).catch(() => [key, null])))
      .then((entries) => alive && setAucs(Object.fromEntries(entries.filter(([, auc]) => auc != null))));
    return () => { alive = false; };
  }, [dataset, availableModels.join(',')]);

  // Load Whisper as soon as the page opens: the first load takes 20-30 s.
  const prepareSpeech = async () => {
    try {
      const status = await getLiveStatus();
      if (!status.installed) { setSpeech({ state: 'missing' }); return; }
      if (status.loaded) { setSpeech({ state: 'ready', ...status }); return; }
      setSpeech({ state: 'loading', ...status });
      setSpeech({ state: 'ready', ...(await loadSpeechModel()) });
    } catch (e) {
      setSpeech({ state: 'error', error: e.message });
    }
  };
  useEffect(() => { prepareSpeech(); }, []);
  useEffect(() => () => captures.current.forEach((capture) => capture.stop()), []);

  // ---- while live ------------------------------------------------------------------------------
  useEffect(() => {
    if (phase !== 'live') return undefined;
    const timer = setInterval(() => { setNow(Date.now()); setLevels({ ...levelRef.current }); }, 120);
    return () => clearInterval(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== 'live' || mode !== 'mic') return undefined;
    const onKey = (event) => {
      if (event.code !== 'Space' || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(event.target.tagName)) return;
      event.preventDefault();
      setSpeaker((current) => (current === 'agent' ? 'customer' : 'agent'));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase, mode]);

  // Re-score whenever the transcript changes. One request at a time; the newest transcript wins.
  useEffect(() => {
    if (phase === 'setup' || !turns.length) return;
    pump.current.wanted = turns.map(({ role, text }) => ({ role, text, speaker: role === 'agent' ? 'Sales rep' : 'Customer' }));
    runPump();
  }, [turnsKey, phase]);

  async function runPump() {
    if (pump.current.busy) return;
    pump.current.busy = true;
    setScoring(true);
    while (pump.current.wanted) {
      const sending = pump.current.wanted;
      pump.current.wanted = null;
      try {
        const out = await scoreLive({ ...selection.current, turns: sending });
        setResult({ ...out, sent: sending.length });
        setScoreError('');
      } catch (e) {
        setScoreError(e.message);
      }
    }
    pump.current.busy = false;
    setScoring(false);
  }

  const addSegment = (segment) => setSegments((current) => [...current, { id: nextId++, ...segment }]);
  const updateSegment = (id, changes) => setSegments((current) => current.map((s) => (s.id === id ? { ...s, ...changes } : s)));

  const listen = (stream, source, role) => new SpeechCapture(stream, {
    onStart: () => (role ?? speakerRef.current),
    onLevel: (rms) => { levelRef.current[role ?? speakerRef.current] = rms; },
    onEnded: () => setNotice(source === 'call' ? 'Call audio sharing stopped, so the customer is no longer heard. End the call or share again from a new call.' : 'The microphone stopped.'),
    onUtterance: ({ audio, startedAt: start, endedAt: end, tag }) => {
      const id = nextId++;
      setSegments((current) => [...current, { id, role: tag, source, start, end, status: 'transcribing', text: '' }]);
      transcribe(audio)
        .then(({ text }) => updateSegment(id, text ? { status: 'done', text } : { status: 'empty' }))
        .catch((e) => { updateSegment(id, { status: 'error' }); setNotice(e.message); });
    },
  });

  const start = async () => {
    setError('');
    setNotice('');
    const opened = [];
    try {
      if (mode === 'call') opened.push(listen(await openCallAudio(), 'call', 'customer')); // needs the click, so it goes first
      if (audioMode) opened.push(listen(await openMicrophone(), 'mic', mode === 'call' ? 'agent' : null));
      for (const capture of opened) await capture.start();
    } catch (e) {
      opened.forEach((capture) => capture.stop());
      setError(e.name === 'NotAllowedError' ? 'Permission was denied. Allow the microphone (and the screen share) and try again.' : e.message);
      return;
    }
    captures.current = opened;
    if (audioMode && speech.state !== 'ready' && speech.state !== 'loading') prepareSpeech();
    setSegments([]);
    setResult(null);
    setScoreError('');
    setPaused(false);
    setSpeaker('agent');
    setStartedAt(Date.now());
    setEndedAt(null);
    setPhase('live');
  };

  const startDemo = () => {
    silence();
    setMode('text');
    setError('');
    setSegments([]);
    setResult(null);
    setStartedAt(Date.now());
    setEndedAt(null);
    setPhase('live');
    setDemo({ step: 0, playing: true });
  };

  // The demo adds one line at a time. With voices on, each line is spoken by its side's voice and
  // fills in word by word, like live captions; the score updates when the line is complete.
  const speaking = voicesOn && voices.supported;
  const playLine = (step) => {
    if (step >= DEMO.length || line.current) return;
    const [role, text] = DEMO[step];
    if (!speaking) {
      addSegment({ role, source: 'demo', start: performance.now(), status: 'done', text });
      setDemo((d) => d && { ...d, step: step + 1 });
      return;
    }
    const id = nextId++;
    setSegments((current) => [...current, { id, role, source: 'demo', start: performance.now(), status: 'transcribing', text: '', partial: '' }]);
    setDemo((d) => d && { ...d, speaking: true });
    const finish = () => {
      line.current = null;
      updateSegment(id, { status: 'done', text, partial: '' });
      setDemo((d) => d && { ...d, step: step + 1, speaking: false });
    };
    // With a single voice in the browser, the two sides still sound different.
    const pitch = voices.sameVoice ? (role === 'agent' ? 0.85 : 1.2) : 1;
    const stop = speakLine(text, { voice: voices[role], pitch, onWord: (partial) => updateSegment(id, { partial }), onDone: finish });
    line.current = { stop, finish };
  };
  /** Ends the line being spoken now, keeping its text. */
  const skipLine = () => { const current = line.current; if (current) { current.stop(); current.finish(); } };
  /** Stops speaking without finishing the line (leaving the page or starting over). */
  function silence() { line.current?.stop(); line.current = null; stopSpeaking(); }
  const demoNext = () => { if (demo?.speaking) skipLine(); else if (demo) playLine(demo.step); };
  const setVoicesOn = (on) => {
    setVoicesOnState(on);
    try { localStorage.setItem(VOICES_KEY, on ? 'on' : 'off'); } catch { /* storage unavailable */ }
    if (!on) skipLine();
  };
  useEffect(() => () => silence(), []);
  useEffect(() => {
    if (!demo?.playing || demo.speaking || phase !== 'live') return undefined;
    if (demo.step >= DEMO.length) { setDemo((d) => d && { ...d, playing: false }); return undefined; }
    const previous = demo.step ? DEMO[demo.step - 1][0] : null;
    const delay = previous ? (speaking ? DEMO_GAP_SPOKEN : DEMO_PAUSE)[previous] : 400;
    const timer = setTimeout(() => playLine(demo.step), delay);
    return () => clearTimeout(timer);
  }, [demo?.step, demo?.playing, demo?.speaking, phase, speaking]);

  const togglePause = () => {
    const next = !paused;
    captures.current.forEach((capture) => capture.setPaused(next));
    setPaused(next);
  };

  const end = () => {
    captures.current.forEach((capture) => capture.stop());
    captures.current = [];
    skipLine();
    setDemo((d) => d && { ...d, playing: false });
    setEndedAt(Date.now());
    setPhase('ended');
  };

  const reset = () => { silence(); setPhase('setup'); setSegments([]); setResult(null); setDemo(null); setNotice(''); setScoreError(''); };

  const flipTurn = (turn) => setSegments((current) => current.map((s) => (turn.ids.includes(s.id) ? { ...s, role: s.role === 'agent' ? 'customer' : 'agent', source: s.source === 'demo' ? 'demo' : 'typed' } : s)));
  const removeTurn = (turn) => setSegments((current) => current.filter((s) => !turn.ids.includes(s.id)));

  const download = () => {
    const text = `# Live call ${new Date(startedAt ?? Date.now()).toLocaleString()}\n${turns.map((t) => `${speakerLabel(t.role, names)}: ${t.text}`).join('\n')}\n`;
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    link.download = `live-call-${new Date(startedAt ?? Date.now()).toISOString().slice(0, 16).replace(/[:T]/g, '-')}.txt`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  if (phase === 'setup') {
    return <Setup {...{ trained, domains, dataset, setDataset, availableModels, model, setModel, aucs, mode, setMode, speech, prepareSpeech, error, start, startDemo, voices, voicesOn, setVoicesOn }} />;
  }

  const elapsed = (endedAt ?? now) - (startedAt ?? now);
  const detail = result?.ready ? result.detail : null;
  const scored = result ? result.sent : 0;
  const offset = result?.dropped_turns ?? 0;
  // The latest result may be one message behind the transcript; it covers the first `scored` turns.
  const scoreAt = (index) => (detail && index >= offset && index < scored ? detail.scores[index - offset] : null);
  const cuesAt = (index) => (result && index >= offset && index < scored ? result.turns[index - offset]?.cues : null);

  return (
    <main className="mx-auto max-w-[1460px] px-5 py-6 md:px-8 md:py-8">
      <ControlBar {...{ phase, mode, paused, togglePause, end, reset, elapsed, levels, speaker, setSpeaker, demo, setDemo, demoNext, voices, voicesOn, setVoicesOn, download, hasTurns: turns.length > 0, names }} />
      {notice && <p role="alert" className="mt-3 flex items-start gap-2 rounded-xl border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100"><TriangleAlert size={15} className="shrink-0" />{notice}</p>}
      {phase === 'live' && audioMode && speech.state !== 'ready' && <SpeechNote speech={speech} retry={prepareSpeech} />}

      <div className="mt-5 grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(380px,.75fr)]">
        <div className="min-w-0 space-y-5">
          <ScoreCard detail={detail} result={result} scoring={scoring} scoreError={scoreError} model={model} dataset={dataset} />
          <Transcript {...{ turns, pending, scoreAt, cuesAt, names, flipTurn, removeTurn, phase, mode, addSegment, echoes, speaker }} />
        </div>
        <div className="xl:sticky xl:top-24">
          <CoachCard coach={result?.coach} detail={detail} dataset={dataset} model={model} phase={phase} hasTurns={turns.length > 0} />
        </div>
      </div>
    </main>
  );
}

// ---- setup ---------------------------------------------------------------------------------------

function Setup({ trained, domains, dataset, setDataset, availableModels, model, setModel, aucs, mode, setMode, speech, prepareSpeech, error, start, startDemo, voices, voicesOn, setVoicesOn }) {
  const audioMode = MODES.find((m) => m.value === mode)?.audio;
  const noModel = trained !== null && domains.length === 0;
  const blocked = noModel || trained === null || (audioMode && ['missing', 'error'].includes(speech.state));
  return (
    <main className="mx-auto max-w-[1080px] px-5 py-8 md:px-8 md:py-12">
      <header className="mx-auto mb-7 max-w-2xl text-center">
        <Badge variant="outline" className="mb-4 border-primary/20 bg-primary/5 text-primary"><Radio size={13} />Live call coach</Badge>
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Coach every call while it happens</h1>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
          Live mode listens to the call, writes the transcript, updates the customer's chance to buy after every message, and suggests what to say next, based on what worked in past conversations.
        </p>
      </header>
      <div className="mx-auto grid max-w-5xl items-start gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,.8fr)]">
        <Card>
          <CardHeader className="border-b border-border/70"><CardTitle>1. How are you taking the call?</CardTitle></CardHeader>
          <CardContent className="space-y-3 p-4 sm:p-6" role="radiogroup" aria-label="Audio setup">
            {MODES.map(({ value, icon: Icon, label, text }) => (
              <button key={value} type="button" role="radio" aria-checked={mode === value} onClick={() => setMode(value)}
                className={`flex w-full items-start gap-3 rounded-2xl border p-4 text-left transition-colors ${mode === value ? 'border-primary bg-primary/5 ring-1 ring-primary/30' : 'border-border hover:border-primary/40'}`}>
                <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl ${mode === value ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}><Icon size={18} /></span>
                <span><span className="block text-sm font-bold">{label}</span><span className="mt-1 block text-xs leading-5 text-muted-foreground">{text}</span></span>
              </button>
            ))}
            <p className="flex items-start gap-2 pt-1 text-[11px] leading-5 text-muted-foreground"><ShieldCheck size={14} className="mt-0.5 shrink-0 text-teal-600" />Speech is transcribed on this computer by Whisper. No audio or transcript is stored or sent anywhere else. Tell the other person the call is being transcribed.</p>
          </CardContent>
        </Card>
        <Card className="lg:sticky lg:top-28">
          <CardHeader><CardTitle>2. Check and start</CardTitle></CardHeader>
          <CardContent className="space-y-5">
            {domains.length > 0 && <div>
              <p className="mb-2 text-xs font-semibold">What kind of conversation is it?</p>
              <Select value={dataset} onValueChange={setDataset}><SelectTrigger aria-label="Conversation type" className="h-10 w-full text-sm"><SelectValue /></SelectTrigger><SelectContent>{domains.map((key) => <SelectItem key={key} value={key}>{DOMAINS[key]}</SelectItem>)}</SelectContent></Select>
              <p className="mt-1 text-[11px] text-muted-foreground">Scores with the model trained on {DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label}, and coaches from its held-out conversations.</p>
              {availableModels.length > 1 && <div className="mt-3">
                <Segmented label="Model" value={model} onChange={setModel}
                  options={availableModels.map((key) => ({ value: key, label: aucs[key] != null ? `${MODELS[key].short} · AUC ${aucs[key].toFixed(2)}` : MODELS[key].label, title: MODELS[key].label }))} />
                <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">{model === 'v3' ? 'v3 re-reads the whole call with a transformer after every message, so updates take a few seconds.' : 'v1 updates in a fraction of a second, which suits a live call.'}</p>
              </div>}
            </div>}
            {noModel && <p role="alert" className="rounded-xl border border-rose-300 bg-rose-50 p-3 text-xs text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">{error || 'No trained model found.'} Live mode needs the data server (`make api`) and a trained model.</p>}
            <SpeechStatus speech={speech} retry={prepareSpeech} needed={audioMode} />
            <div className="space-y-2">
              <Button type="button" className="w-full rounded-xl" disabled={blocked} onClick={start}><Radio size={16} />{mode === 'text' ? 'Start typing a call' : 'Start listening'}</Button>
              <Button type="button" variant="outline" className="w-full rounded-xl" disabled={noModel || trained === null} onClick={startDemo}><Play size={15} />Play a demo call</Button>
              {voices.supported
                ? <label className="flex cursor-pointer items-start gap-2 pt-1 text-[11px] leading-4 text-muted-foreground">
                    <input type="checkbox" checked={voicesOn} onChange={(event) => setVoicesOn(event.target.checked)} className="mt-0.5 h-3.5 w-3.5 accent-blue-600" />
                    <span>Speak the demo aloud{voicesOn && voices.agent && <> · sales rep: <b>{shortName(voices.agent)}</b>, customer: <b>{shortName(voices.customer)}</b>{voices.sameVoice && ' (one voice, two pitches)'}</>}</span>
                  </label>
                : <p className="pt-1 text-[11px] text-muted-foreground">This browser has no speech voices, so the demo plays silently.</p>}
            </div>
            {error && !noModel && <p role="alert" className="rounded-xl border border-rose-300 bg-rose-50 p-3 text-xs text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">{error}</p>}
            {mode === 'call' && <p className="text-[11px] leading-5 text-muted-foreground">After you click Start, choose the call's tab and tick <b>Also share tab audio</b>. For a desktop app (Zoom, Teams), share your entire screen with <b>Share system audio</b>.</p>}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

function SpeechStatus({ speech, retry, needed }) {
  const base = 'flex items-start gap-2 rounded-xl border p-3 text-xs leading-5';
  if (speech.state === 'ready') return <p className={`${base} border-teal-200 bg-teal-50 text-teal-900 dark:border-teal-900 dark:bg-teal-950/30 dark:text-teal-100`}><Check size={15} className="mt-0.5 shrink-0" />Speech model ready ({speech.model?.split('/').pop()} on {speech.device === 'cuda' ? 'the GPU' : 'the CPU'}).</p>;
  if (speech.state === 'checking' || speech.state === 'loading') return <p className={`${base} border-border bg-muted/40 text-muted-foreground`}><LoaderCircle size={15} className="mt-0.5 shrink-0 animate-spin" />{speech.state === 'checking' ? 'Checking the speech model…' : 'Loading the speech model. This takes 20–30 seconds (longer the very first time, while it downloads).'}</p>;
  if (speech.state === 'missing') return <p className={`${base} ${needed ? 'border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100' : 'border-border bg-muted/40 text-muted-foreground'}`}><TriangleAlert size={15} className="mt-0.5 shrink-0" />Speech-to-text needs torch and transformers (requirements-ml.txt). Typing and the demo still work.</p>;
  return <div className={`${base} ${needed ? 'border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100' : 'border-border bg-muted/40 text-muted-foreground'}`}><TriangleAlert size={15} className="mt-0.5 shrink-0" /><span>The speech model could not be loaded: {speech.error}<Button type="button" variant="link" size="sm" className="h-auto px-1 text-xs" onClick={retry}>Try again</Button></span></div>;
}

function SpeechNote({ speech, retry }) {
  return <div className="mt-3"><SpeechStatus speech={speech} retry={retry} needed /></div>;
}

// ---- live view -----------------------------------------------------------------------------------

function Meter({ label, level, active }) {
  const width = Math.min(100, Math.round(Math.sqrt(level) * 260));
  return (
    <div className="min-w-28">
      <p className={`text-[10px] font-bold uppercase tracking-wider ${active ? 'text-primary' : 'text-muted-foreground'}`}>{label}</p>
      <div className="mt-1 h-1.5 w-28 overflow-hidden rounded-full bg-muted" aria-hidden="true"><div className="h-full rounded-full bg-primary transition-[width] duration-100" style={{ width: `${width}%` }} /></div>
    </div>
  );
}

function ControlBar({ phase, mode, paused, togglePause, end, reset, elapsed, levels, speaker, setSpeaker, demo, setDemo, demoNext, voices, voicesOn, setVoicesOn, download, hasTurns, names }) {
  const live = phase === 'live';
  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-x-5 gap-y-3 p-4">
        <span className="inline-flex items-center gap-2 text-sm font-extrabold">
          <span className={`h-2.5 w-2.5 rounded-full ${live && !paused ? 'animate-pulse bg-rose-500' : 'bg-muted-foreground/50'}`} />
          {live ? (paused ? 'Paused' : demo ? 'Demo call' : 'Live') : 'Call ended'}
          <span className="font-mono text-xs font-semibold tabular-nums text-muted-foreground">{clock(elapsed)}</span>
        </span>
        {live && mode === 'call' && <><Meter label="You" level={levels.agent} /><Meter label={names.customer} level={levels.customer} /></>}
        {live && mode === 'mic' && <>
          <Meter label={speaker === 'agent' ? 'You' : names.customer} level={levels[speaker]} active />
          <Button type="button" variant="outline" size="sm" onClick={() => setSpeaker(speaker === 'agent' ? 'customer' : 'agent')} aria-label="Switch who is talking (Space)">
            <Repeat2 size={14} />Now talking: <b>{speaker === 'agent' ? 'You' : names.customer}</b><kbd className="ml-1 rounded border border-border px-1 text-[10px] text-muted-foreground">Space</kbd>
          </Button>
        </>}
        {live && demo && <>
          <span className="text-xs tabular-nums text-muted-foreground">Line {Math.min(demo.step + (demo.speaking ? 1 : 0), DEMO.length)} of {DEMO.length}</span>
          <Button type="button" variant="outline" size="sm" onClick={() => setDemo({ ...demo, playing: !demo.playing })} disabled={demo.step >= DEMO.length}>{demo.playing ? <><Pause size={14} />Pause demo</> : <><Play size={14} />Resume</>}</Button>
          <Button type="button" variant="ghost" size="sm" onClick={demoNext} disabled={demo.step >= DEMO.length}><SkipForward size={14} />{demo.speaking ? 'Skip line' : 'Next line'}</Button>
          {voices.supported && <Button type="button" variant="ghost" size="sm" onClick={() => setVoicesOn(!voicesOn)} aria-pressed={voicesOn} title={voicesOn ? 'Mute the demo voices' : 'Speak the demo aloud'}>
            {voicesOn ? <Volume2 size={14} /> : <VolumeX size={14} />}{voicesOn ? 'Voices on' : 'Voices off'}
          </Button>}
        </>}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {live && mode !== 'text' && <Button type="button" variant="outline" size="sm" onClick={togglePause}>{paused ? <><Play size={14} />Resume</> : <><Pause size={14} />Pause</>}</Button>}
          {live && <Button type="button" size="sm" className="bg-rose-600 text-white hover:bg-rose-700" onClick={end}><PhoneOff size={14} />End call</Button>}
          {!live && <><Button type="button" variant="outline" size="sm" onClick={download} disabled={!hasTurns}><Download size={14} />Download transcript</Button><Button type="button" size="sm" onClick={reset}><RotateCcw size={14} />New call</Button></>}
        </div>
      </CardContent>
    </Card>
  );
}

function ChartHint({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return <div className="rounded-xl border border-border bg-card px-3 py-2 text-xs shadow-float"><p className="font-bold">Message {point.turn + 1} · {point.role === 'agent' ? 'You' : 'Customer'}</p><p className="mt-1 tabular-nums text-primary">Chance to buy: {pct(point.score)}</p></div>;
}

function ScoreCard({ detail, result, scoring, scoreError, model, dataset }) {
  const scores = detail?.scores ?? [];
  const score = scores.at(-1);
  const previous = scores.length > 1 ? scores.at(-2) : null;
  const data = scores.map((value, index) => ({ turn: index, score: value, role: detail.turns[index]?.role }));
  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3 border-b border-border/70 pb-4">
        <div>
          <CardTitle className="flex items-center gap-2">Chance to buy, live {scoring && <LoaderCircle size={14} className="animate-spin text-muted-foreground" aria-label="Updating" />}</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">{MODELS[model]?.label ?? model}, trained on {DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label}. Updated after every message.</p>
        </div>
        {detail && <div className="flex items-center gap-2">
          <span className={`text-3xl font-extrabold tabular-nums ${score >= 0.7 ? 'text-orange-600 dark:text-orange-300' : 'text-primary'}`}>{pct(score)}</span>
          {previous != null && <Badge variant="outline" className={score >= previous ? 'border-teal-200 text-teal-800 dark:border-teal-900 dark:text-teal-200' : 'border-orange-200 text-orange-800 dark:border-orange-900 dark:text-orange-200'}>{score >= previous ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}{pts(score - previous)}</Badge>}
          <StatusTag status={detail.status} />
        </div>}
      </CardHeader>
      <CardContent className="px-2 pb-3 pt-4 sm:px-5">
        {scoreError && <p role="alert" className="mx-2 mb-3 rounded-xl border border-rose-300 bg-rose-50 p-3 text-xs text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">Scoring failed: {scoreError}</p>}
        {!detail ? (
          <div className="grid h-[200px] place-items-center rounded-2xl border border-dashed border-border text-center text-sm text-muted-foreground"><p className="max-w-sm px-4">{result?.message ?? 'Start talking. The score appears once both you and the customer have spoken.'}</p></div>
        ) : (
          <>
            <div className="h-[220px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 12, right: 14, left: 0, bottom: 2 }}>
                  <defs><linearGradient id="live-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2563eb" stopOpacity={0.28} /><stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} /></linearGradient></defs>
                  <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="4 5" />
                  <XAxis dataKey="turn" tickFormatter={(v) => v + 1} tickLine={false} axisLine={false} fontSize={11} tick={{ fill: 'hsl(var(--muted-foreground))' }} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} fontSize={11} tick={{ fill: 'hsl(var(--muted-foreground))' }} width={42} />
                  <ReferenceLine y={0.7} stroke="#0f9f88" strokeDasharray="5 5" strokeOpacity={0.65} />
                  <ChartTooltip content={<ChartHint />} />
                  <Area type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2.5} fill="url(#live-fill)" isAnimationActive={false}
                    dot={(props) => <circle key={`dot-${props.payload.turn}`} cx={props.cx} cy={props.cy} r={props.payload.role === 'customer' ? 3.2 : 1.8} fill={props.payload.role === 'customer' ? '#2563eb' : 'hsl(var(--card))'} stroke="#2563eb" strokeWidth={1.5} />} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 px-2 text-[11px] text-muted-foreground">{cap(detail.status_reason)}. Filled dots are customer messages; the dashed line marks a hot lead (70%).{result.dropped_turns > 0 && ` Scored on the latest ${detail.n_turns} messages.`}</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function CueChips({ cues }) {
  if (!cues?.length) return null;
  return <div className="mt-1.5 flex flex-wrap gap-1">{cues.map((cue) => {
    const direction = CUSTOMER_CUES[cue]?.direction;
    return <Badge key={cue} variant="outline" className={`border-0 px-1.5 py-0 text-[10px] ${direction > 0 ? 'bg-teal-50 text-teal-800 dark:bg-teal-950/50 dark:text-teal-100' : direction < 0 ? 'bg-orange-50 text-orange-800 dark:bg-orange-950/50 dark:text-orange-100' : 'bg-muted text-muted-foreground'}`}>{pretty(cue)}</Badge>;
  })}</div>;
}

function Transcript({ turns, pending, scoreAt, cuesAt, names, flipTurn, removeTurn, phase, mode, addSegment, echoes, speaker }) {
  const box = useRef(null);
  const following = useRef(true); // keep the newest message in view unless the user scrolled up
  const [draft, setDraft] = useState('');
  const [draftRole, setDraftRole] = useState('agent');
  useEffect(() => {
    if (box.current && following.current) box.current.scrollTo({ top: box.current.scrollHeight, behavior: 'smooth' });
  }, [turns.length, pending.length, pending.map((segment) => segment.partial?.length ?? 0).join()]);
  const send = (event) => {
    event.preventDefault();
    if (!draft.trim()) return;
    addSegment({ role: draftRole, source: 'typed', start: performance.now(), status: 'done', text: draft.trim() });
    setDraft('');
    setDraftRole(draftRole === 'agent' ? 'customer' : 'agent');
  };
  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border/70 pb-4">
        <CardTitle>Transcript</CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">Click a speaker name to swap sides if a message was put on the wrong side.{echoes > 0 && ` ${echoes} echo${echoes === 1 ? '' : 'es'} of the customer picked up by your microphone ${echoes === 1 ? 'was' : 'were'} removed.`}</p>
      </CardHeader>
      <div ref={box} onScroll={(event) => { const el = event.currentTarget; following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}
        className="max-h-[560px] space-y-2.5 overflow-y-auto p-4 sm:p-5" aria-live="polite">
        {turns.length === 0 && pending.length === 0 && <p className="rounded-xl border border-dashed border-border p-5 text-center text-sm text-muted-foreground">{mode === 'text' ? 'Type the first message below.' : mode === 'mic' ? `Start talking. Press Space when the other person speaks (now: ${speaker === 'agent' ? 'you' : names.customer.toLowerCase()}).` : 'Start talking. Messages appear here a moment after each sentence.'}</p>}
        {turns.map((turn, index) => {
          const score = scoreAt(index);
          const previous = index > 0 ? scoreAt(index - 1) : null;
          const delta = score != null && previous != null ? score - previous : null;
          const customer = turn.role === 'customer';
          return (
            <div key={turn.ids[0]} className={`group flex gap-2.5 ${customer ? 'flex-row-reverse' : ''}`}>
              <div className={`min-w-0 max-w-[88%] rounded-2xl border px-3.5 py-2.5 shadow-sm ${customer ? 'border-blue-100 bg-blue-50/70 dark:border-blue-900/50 dark:bg-blue-950/30' : 'border-border bg-card'}`}>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <button type="button" onClick={() => flipTurn(turn)} className="text-[11px] font-bold hover:text-primary hover:underline" title="Swap to the other side">{customer ? names.customer : 'You'}</button>
                  {score != null && <Badge variant="outline" className="h-5 px-1.5 text-[10px] tabular-nums">{pct(score)}</Badge>}
                  {delta != null && Math.abs(delta) >= 0.01 && <span className={`text-[10px] font-bold ${delta > 0 ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}`}>{pts(delta)}</span>}
                  {phase === 'live' && <button type="button" onClick={() => removeTurn(turn)} className="ml-auto text-muted-foreground opacity-0 transition-opacity hover:text-rose-600 focus:opacity-100 group-hover:opacity-100" aria-label="Remove this message"><Trash2 size={13} /></button>}
                </div>
                <p className="mt-1 whitespace-pre-wrap text-sm leading-6">{turn.text}</p>
                <CueChips cues={cuesAt(index)} />
              </div>
            </div>
          );
        })}
        {pending.map((segment) => {
          const who = segment.role === 'customer' ? names.customer : 'You';
          if (segment.source === 'demo') {
            return (
              <div key={segment.id} className={`flex ${segment.role === 'customer' ? 'flex-row-reverse' : ''}`}>
                <div className={`max-w-[88%] rounded-2xl border border-dashed px-3.5 py-2.5 ${segment.role === 'customer' ? 'border-blue-200 bg-blue-50/40 dark:border-blue-900/60 dark:bg-blue-950/20' : 'border-border bg-card/60'}`}>
                  <p className="flex items-center gap-1.5 text-[11px] font-bold"><AudioLines size={13} className="animate-pulse text-primary" />{who}<span className="font-normal text-muted-foreground">speaking…</span></p>
                  {segment.partial && <p className="mt-1 text-sm leading-6">{segment.partial}<span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-pulse bg-primary" /></p>}
                </div>
              </div>
            );
          }
          return (
            <div key={segment.id} className={`flex ${segment.role === 'customer' ? 'flex-row-reverse' : ''}`}>
              <span className="inline-flex items-center gap-2 rounded-2xl border border-dashed border-border px-3.5 py-2 text-xs text-muted-foreground"><LoaderCircle size={12} className="animate-spin" />{who}: transcribing…</span>
            </div>
          );
        })}
      </div>
      {phase === 'live' && <form onSubmit={send} className="flex flex-wrap items-center gap-2 border-t border-border/70 p-3">
        <Segmented label="Type as" value={draftRole} onChange={setDraftRole} options={[{ value: 'agent', label: 'You' }, { value: 'customer', label: names.customer }]} />
        <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={mode === 'text' ? 'Type a message…' : 'Add or correct a message…'} aria-label="Message text"
          className="h-9 min-w-48 flex-1 rounded-xl border border-input bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-ring" />
        <Button type="submit" size="sm" disabled={!draft.trim()}><Send size={14} />Add</Button>
      </form>}
    </Card>
  );
}

function CoachCard({ coach, detail, dataset, model, phase, hasTurns }) {
  const lastPoint = detail?.turning_points?.at(-1);
  if (!coach) {
    return (
      <Card><CardHeader><CardTitle className="flex items-center gap-2"><Lightbulb size={17} className="text-amber-500" />What to say next</CardTitle></CardHeader>
        <CardContent><p className="rounded-xl border border-dashed border-border p-5 text-sm text-muted-foreground">{hasTurns ? 'Reading the conversation…' : 'Suggestions appear here as soon as the conversation starts.'}</p></CardContent></Card>
    );
  }
  const names = DATASETS[dataset] ?? { customer: 'Customer' };
  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border/70 pb-4">
        <CardTitle className="flex items-center gap-2"><Lightbulb size={17} className="text-amber-500" />{phase === 'ended' ? 'Coaching at the end of the call' : 'What to say next'}</CardTitle>
        <p className="mt-2 text-base font-bold leading-6">{coach.headline}</p>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Badge variant="secondary">{coach.situation_label}</Badge>
          {coach.cues.filter((cue) => cue !== coach.situation).map((cue) => <Badge key={cue} variant="outline" className="text-[10px]">{pretty(cue)}</Badge>)}
        </div>
        {coach.quote && <p className="mt-2 flex gap-1.5 text-xs italic leading-5 text-muted-foreground"><Quote size={12} className="mt-1 shrink-0" />{coach.quote}</p>}
      </CardHeader>
      <CardContent className="space-y-4 p-4 sm:p-5">
        {coach.waiting_for === 'customer' && coach.suggestions.length > 0 && <p className="rounded-xl bg-muted/60 px-3 py-2 text-xs text-muted-foreground">Waiting for the {names.customer.toLowerCase()}'s reply. These suggestions answer their last message.</p>}
        {coach.suggestions.length === 0 && <p className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">No sales move has a clear record in this situation yet. Keep the customer talking.</p>}
        {coach.suggestions.map((suggestion, index) => <Suggestion key={suggestion.move} suggestion={suggestion} first={index === 0} names={names} />)}
        {coach.avoid.length > 0 && <div className="rounded-2xl border border-rose-200 bg-rose-50/60 p-3.5 dark:border-rose-900/60 dark:bg-rose-950/20">
          <p className="mb-2 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-rose-800 dark:text-rose-200"><Ban size={13} />Avoid right now</p>
          <ul className="space-y-2">{coach.avoid.map((item) => <li key={item.move} className="text-xs leading-5"><b>{item.action}</b>{item.just_used && <Badge variant="outline" className="ml-1.5 border-rose-300 px-1.5 py-0 text-[10px] text-rose-800 dark:text-rose-200">you just did this</Badge>}<span className="block text-muted-foreground">{item.why}</span></li>)}</ul>
        </div>}
        {lastPoint && <div className={`rounded-2xl border p-3.5 text-xs leading-5 ${lastPoint.direction === 'rise' ? 'border-amber-200 bg-amber-50/70 dark:border-amber-900/60 dark:bg-amber-950/20' : 'border-sky-200 bg-sky-50/70 dark:border-sky-900/60 dark:bg-sky-950/20'}`}>
          <p className="mb-1 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider">{lastPoint.direction === 'rise' ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}Latest turning point · message {lastPoint.turn + 1} · {pts(lastPoint.delta)}</p>
          <p>{lastPoint.rationale}</p>
        </div>}
        <p className="text-[11px] leading-5 text-muted-foreground">
          {coach.n_conversations > 0
            ? <>Based on how the score moved after each kind of reply in {coach.n_conversations.toLocaleString()} held-out {DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label} conversations scored by {MODELS[model]?.short ?? model}{coach.evidence_from === 'any' && coach.situation !== 'any' ? ' (this situation is rare there, so every reply is used)' : ''}. These are patterns in past conversations, not guarantees. Fill in the [brackets] with your specifics.</>
            : 'The coaching playbook needs the exported outputs (`make export`); only the headline is available.'}
        </p>
      </CardContent>
    </Card>
  );
}

function Suggestion({ suggestion, first, names }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const evidence = EVIDENCE[suggestion.evidence] ?? EVIDENCE.typical;
  const copy = async () => {
    try { await navigator.clipboard.writeText(suggestion.say); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard unavailable */ }
  };
  return (
    <div className={`rounded-2xl border p-3.5 ${first ? 'border-primary/40 bg-primary/5' : 'border-border'}`}>
      <div className="flex flex-wrap items-center gap-2">
        {first && <Sparkles size={14} className="text-primary" />}
        <p className="text-sm font-bold">{suggestion.action}</p>
        <Badge variant="outline" className={`px-1.5 py-0 text-[10px] ${evidence.className}`}>{evidence.label}</Badge>
      </div>
      <div className="mt-2 flex items-start gap-2 rounded-xl bg-card/90 p-2.5 shadow-sm ring-1 ring-border/60">
        <p className="flex-1 text-sm leading-6">“{suggestion.say}”</p>
        <Button type="button" variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={copy} aria-label="Copy this line">{copied ? <Check size={13} /> : <Copy size={13} />}</Button>
      </div>
      <p className="mt-2 text-[11px] leading-5 text-muted-foreground">{suggestion.why}</p>
      {suggestion.examples.length > 0 && <>
        <button type="button" onClick={() => setOpen(!open)} className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-semibold text-primary hover:underline" aria-expanded={open}>
          <ChevronDown size={12} className={`transition-transform ${open ? 'rotate-180' : ''}`} />{open ? 'Hide' : 'Show'} what this looked like in real conversations
        </button>
        {open && <div className="mt-2 space-y-2">{suggestion.examples.map((example, index) => (
          <div key={index} className="space-y-1 rounded-xl border border-border/70 bg-muted/30 p-2.5 text-[11px] leading-5">
            {example.customer && <p><b>{names.customer}:</b> {example.customer}</p>}
            <p><b>{names.agent ?? 'Sales rep'}:</b> {example.agent}</p>
            <p><b>{names.customer}:</b> {example.reply} <span className="font-bold text-teal-700 dark:text-teal-200">{pts(example.delta)}</span></p>
          </div>
        ))}</div>}
      </>}
    </div>
  );
}
