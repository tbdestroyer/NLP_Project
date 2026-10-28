import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, BadgeCheck, FileUp, Files, LoaderCircle, MessageSquareText, Sparkles, Trash2, Upload } from 'lucide-react';
import { analyzeBatch, analyzeConversation, getAnalyzeModels, getMetrics } from '../api/client';
import BatchResults from '../components/BatchResults';
import ConversationDetail from '../components/ConversationDetail';
import { Segmented } from '../components/ui';
import { DATASETS, MODELS } from '../data/taxonomy';
import { hasBothSides, parseConversation, parseTranscripts, roleGuesser, withRoles } from '../data/transcripts';
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/primitives';

const SAMPLE = `Rep: Hi Dana, thanks for making time today. How are you?
Dana: Good, thanks. We're looking at tools for our support team.
Rep: Great. How many seats would you need?
Dana: Around 40. How much is the team plan?
Rep: It's $40 per seat per month, plus a setup fee.
Dana: Hmm, that's out of our budget for this quarter.
Rep: I understand. I can offer 15% off for an annual plan, and the setup fee goes away.
Dana: Okay, that works better. Does it integrate with Salesforce?
Rep: Yes, natively. Customers like Contoso cut response time 30% in the first month.
Dana: That's impressive. What are the next steps for onboarding?
Rep: Shall we send the contract over today?
Dana: Sounds good, send it over.`;

const SAMPLE_BATCH = `# Northwind support team
${SAMPLE}
---
# Fabrikam analytics trial
Rep: Hi Omar, how did the first week of the analytics trial go?
Omar: Honestly, the team barely logged in. We've been busy.
Rep: That's common in week one. What would make it worth their time?
Omar: Not sure. We're also looking at Tableau.
Rep: Tableau is strong. Our edge is setup in a day and a lower price per seat.
Omar: Maybe. I'll need to think about it and get back to you next quarter.
Rep: Of course. Shall I check in in January?
Omar: Fine.
---
# Litware onboarding suite
Customer: Hey, I saw your post about the onboarding suite. Our process takes forever.
Sales rep: Totally get that. What slows new hires down the most?
Customer: Paperwork and getting accounts set up. How does your tool help?
Sales rep: Checklists and automatic account setup. Teams cut onboarding time about 50%.
Customer: That sounds great. Does it work with Workday?
Sales rep: Yes, there's a native Workday integration.
Customer: Perfect. What's the pricing for 200 employees?
Sales rep: $299 a month on the annual plan, with a 30-day free trial.
Customer: Let's do the trial. How do we get started?
---
# Adatum security add-on
Agent: Hello Priya, following up on the security add-on quote.
Priya: Thanks. The price is a lot higher than we expected.
Agent: It includes 24/7 monitoring, which usually costs extra elsewhere.
Priya: I don't trust that the monitoring is better than what we have.
Agent: I can share a case study from a customer your size.
Priya: No thanks, we're not interested right now.
---
# Tailspin CRM upgrade
Rep: Hi Lee, any questions after the CRM demo?
Lee: Yes, how long does migration take?
Rep: Usually two weeks, and we handle the data import for you.
Lee: Good. And the price?
Rep: $55 per user per month. This month there's 10% off if you sign before Friday.
Lee: That's a bit expensive. Can you do better on annual billing?
Rep: Annual billing brings it to $48 per user.
Lee: Okay, that works. Send me the order form.`;

// Each trained model learned from one kind of conversation; the user picks the closest one.
const DOMAINS = {
  saas: 'Sales call or chat',
  craigslist: 'Price negotiation',
  p4g: 'Persuasion or donation ask',
};
const PREFERRED_DOMAIN = 'saas';
const MAX_FILE_BYTES = 10_000_000;
const MAX_CONVERSATIONS = 300; // config.yaml analyze.max_conversations

// Results survive leaving the page (they are never sent anywhere or stored on disk).
const memory = { mode: 'single', single: null, batch: null };

const PreviewNote = ({ children }) => <div className="mb-5 flex items-start gap-3 rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-950 dark:border-blue-900 dark:bg-blue-950/30 dark:text-blue-100"><BadgeCheck size={17} className="mt-0.5 shrink-0" /><span>{children}</span></div>;
const Alert = ({ children }) => <div role="alert" className="mt-3 rounded-xl border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">{children}</div>;
const Warning = ({ children }) => <p role="alert" className="mt-3 rounded-xl border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100">{children}</p>;

export default function Analyze() {
  const [mode, setModeState] = useState(memory.mode); // single | batch
  const [trained, setTrained] = useState(null); // { [dataset]: [models] } from the server
  const [trainedError, setTrainedError] = useState('');
  const [aucs, setAucs] = useState({}); // { [model]: test ROC AUC } for the chosen dataset
  const [dataset, setDataset] = useState(PREFERRED_DOMAIN);
  const domains = Object.keys(DOMAINS).filter((key) => trained?.[key]?.length);
  const preview = trained !== null && domains.length === 0;
  const availableModels = (trained?.[dataset] ?? []).filter((key) => key in MODELS);
  const [model, setModel] = useState('v1');
  const [modelChosen, setModelChosen] = useState(false);

  const [raw, setRaw] = useState('');
  const [title, setTitle] = useState('');
  const [inputTab, setInputTab] = useState('paste');
  const [roles, setRoles] = useState({}); // speaker -> 'agent' | 'customer'
  const [over, setOver] = useState(false);
  const [status, setStatus] = useState(memory[memory.mode] ? 'done' : 'idle'); // idle | running | done | error
  const [result, setResult] = useState(memory.single);
  const [batchResult, setBatchResult] = useState(memory.batch);
  const [error, setError] = useState('');
  const [files, setFiles] = useState([]); // [{ name, conversations }]
  const [batchText, setBatchText] = useState('');
  const fileInput = useRef(null);

  const setMode = (next) => {
    memory.mode = next;
    setModeState(next);
    setError('');
    setStatus(memory[next] ? 'done' : 'idle');
  };

  // One conversation
  const turns = useMemo(() => parseConversation(raw), [raw]);
  const speakers = useMemo(() => [...new Set(turns.map((t) => t.speaker))], [turns]);
  const roleOf = roleGuesser(turns, roles);
  const bothSides = hasBothSides(withRoles(turns, roles));

  // Many conversations
  const conversations = useMemo(() => [
    ...files.flatMap((file) => file.conversations),
    ...parseTranscripts(batchText, 'Pasted conversation'),
  ], [files, batchText]);
  const prepared = useMemo(() => conversations.map((conversation) => ({ title: conversation.title, turns: withRoles(conversation.turns, roles) })), [conversations, roles]);
  const unusable = prepared.filter((conversation) => conversation.turns.length < 2 || !hasBothSides(conversation.turns));
  const batchSpeakers = useMemo(() => {
    const found = new Map();
    for (const conversation of conversations) {
      const guess = roleGuesser(conversation.turns, roles);
      for (const speaker of new Set(conversation.turns.map((turn) => turn.speaker))) {
        const entry = found.get(speaker) ?? { speaker, role: guess(speaker), chats: 0 };
        entry.chats += 1;
        found.set(speaker, entry);
      }
    }
    return [...found.values()].sort((a, b) => b.chats - a.chats || a.speaker.localeCompare(b.speaker));
  }, [conversations, roles]);

  useEffect(() => {
    getAnalyzeModels().then(setTrained).catch((e) => { setTrainedError(e.message); setTrained({}); });
  }, []);
  useEffect(() => {
    if (domains.length && !domains.includes(dataset)) setDataset(domains[0]);
  }, [domains.join(','), dataset]);
  useEffect(() => {
    let alive = true;
    setAucs({});
    Promise.all(availableModels.map((key) => getMetrics({ dataset, model: key }).then((m) => [key, m?.roc_auc]).catch(() => [key, null])))
      .then((entries) => alive && setAucs(Object.fromEntries(entries.filter(([, auc]) => auc != null))));
    return () => { alive = false; };
  }, [dataset, availableModels.join(',')]);
  // Default to the model that ranks this kind of conversation best on its held-out test split.
  const bestModel = availableModels.reduce((best, key) => ((aucs[key] ?? -1) > (aucs[best] ?? -1) ? key : best), availableModels[0]);
  useEffect(() => {
    if (!availableModels.length) return;
    if (!availableModels.includes(model) || (!modelChosen && bestModel && model !== bestModel)) setModel(bestModel ?? availableModels[0]);
  }, [availableModels.join(','), model, bestModel, modelChosen]);

  const readText = async (file, maxBytes) => {
    const extension = file.name.toLowerCase().split('.').pop();
    if (!['txt', 'csv', 'json'].includes(extension)) throw new Error(`${file.name}: choose .txt, .csv or .json files.`);
    if (file.size > maxBytes) throw new Error(`${file.name} is over ${maxBytes / 1_000_000} MB.`);
    return file.text();
  };

  const loadFile = async (file) => {
    if (!file) return;
    setError('');
    try {
      setRaw(await readText(file, 2_000_000));
    } catch (readError) {
      setError(readError.message);
      return;
    }
    setInputTab('paste');
    if (!title) setTitle(file.name.replace(/\.[^.]+$/, ''));
  };

  const loadFiles = async (list) => {
    const chosen = [...(list ?? [])];
    if (!chosen.length) return;
    setError('');
    const loaded = [];
    const problems = [];
    for (const file of chosen) {
      try {
        const found = parseTranscripts(await readText(file, MAX_FILE_BYTES), file.name.replace(/\.[^.]+$/, ''));
        if (found.length) loaded.push({ name: file.name, conversations: found });
        else problems.push(`${file.name}: no messages found.`);
      } catch (readError) {
        problems.push(readError.message);
      }
    }
    setFiles((current) => [...current.filter((file) => !loaded.some((item) => item.name === file.name)), ...loaded]);
    if (problems.length) setError(problems.join(' '));
  };

  const run = async () => {
    setStatus('running');
    setError('');
    try {
      const out = await analyzeConversation({ title, dataset, model, preview, turns: withRoles(turns, roles) });
      memory.single = out;
      setResult(out);
      setStatus('done');
    } catch (e) {
      setError(e.message);
      setStatus('error');
    }
  };

  const runBatch = async () => {
    setStatus('running');
    setError('');
    try {
      const usable = prepared.filter((conversation) => !unusable.includes(conversation));
      const out = await analyzeBatch({ dataset, model, conversations: usable });
      const skippedHere = prepared.filter((conversation) => unusable.includes(conversation))
        .map((conversation) => ({ index: -1, title: conversation.title, reason: 'Needs at least two messages, with a customer and a sales side.' }));
      const merged = { ...out, skipped: [...skippedHere, ...out.skipped] };
      memory.batch = merged;
      setBatchResult(merged);
      setStatus('done');
    } catch (e) {
      setError(e.message);
      setStatus('error');
    }
  };

  const resetSingle = () => { memory.single = null; setRaw(''); setTitle(''); setRoles({}); setResult(null); setStatus('idle'); setInputTab('paste'); };
  const resetBatch = () => { memory.batch = null; setFiles([]); setBatchText(''); setRoles({}); setBatchResult(null); setStatus('idle'); };

  if (status === 'done' && mode === 'batch' && batchResult) {
    return <BatchResults result={batchResult} onEdit={() => setStatus('idle')} onReset={resetBatch} />;
  }
  if (status === 'done' && mode === 'single' && result) {
    return (
      <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
        <div className="mb-5 flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => setStatus('idle')}><ArrowLeft size={15} />Edit conversation</Button>
          <Button variant="ghost" size="sm" onClick={resetSingle}>Analyze another</Button>
        </div>
        {result.model === 'preview'
          ? <PreviewNote><b>Preview: no trained model was available.</b><br />These numbers come from simple keyword rules in the browser, not from the team's models.</PreviewNote>
          : <PreviewNote><b>Scored by {MODELS[result.model]?.label ?? result.model}, trained on{DATASETS[result.dataset]?.displayLabel ?? DATASETS[result.dataset]?.label ?? result.dataset}.</b><br />Models trained on one kind of conversation transfer poorly to others, so treat the score as a rough guide.</PreviewNote>}
        <ConversationDetail convo={result} model={result.model} split={null} />
      </main>
    );
  }

  const running = status === 'running';
  const batchCount = prepared.length - unusable.length;
  const tooMany = batchCount > MAX_CONVERSATIONS;
  const modelPicker = domains.length > 0 && (
    <div><p className="mb-2 text-xs font-semibold">{mode === 'batch' ? '2' : '3'}. What kind of conversation is it?</p>
      <Select value={dataset} onValueChange={(value) => { setDataset(value); setModelChosen(false); }}><SelectTrigger aria-label="Conversation type" className="h-10 w-full text-sm"><SelectValue /></SelectTrigger><SelectContent>{domains.map((key) => <SelectItem key={key} value={key}>{DOMAINS[key]}</SelectItem>)}</SelectContent></Select>
      <p className="mt-1 text-[11px] text-muted-foreground">Uses the model trained on {DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label}.</p>
      {availableModels.length > 1 && <div className="mt-3">
        <Segmented label="Model" value={model} onChange={(value) => { setModel(value); setModelChosen(true); }}
          options={availableModels.map((key) => ({ value: key, label: aucs[key] != null ? `${MODELS[key].short} · AUC ${aucs[key].toFixed(2)}` : MODELS[key].label, title: MODELS[key].label }))} />
        {bestModel && aucs[bestModel] != null && <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">
          {MODELS[bestModel].label} ranks {DOMAINS[dataset].toLowerCase()}s best on held-out chats (ROC AUC {aucs[bestModel].toFixed(2)}){model === bestModel ? ', so it is selected.' : '.'}
          {model === 'v3' && ' v3 runs a transformer on every message, so large batches take longer.'}
        </p>}
      </div>}
    </div>
  );

  return (
    <main className="mx-auto max-w-[1080px] px-5 py-8 md:px-8 md:py-12">
      <header className="mx-auto mb-6 max-w-2xl text-center">
        <Badge variant="outline" className="mb-4 border-primary/20 bg-primary/5 text-primary"><Sparkles size={13} />Analyze conversations</Badge>
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Read the temperature of your conversations</h1>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-muted-foreground">
          {mode === 'batch'
            ? 'Upload many sales conversations to rank them as a lead list, see which sales moves help or hurt, and open any chat for its full score curve.'
            : "Add a conversation to see its score after each message, likely turning points, and what may have changed the customer's mind."}
        </p>
        <div className="mt-5 flex justify-center">
          <Segmented label="How many conversations" value={mode} onChange={setMode}
            options={[{ value: 'single', label: 'One conversation' }, { value: 'batch', label: 'Many conversations', disabled: preview, title: preview ? 'Needs a trained model' : undefined }]} />
        </div>
      </header>
      {preview && <div className="mx-auto max-w-3xl"><PreviewNote><b>Preview: no trained model found.</b><br />{trainedError ? `The data server did not answer (${trainedError}). Start it with \`make api\`.` : 'Run `make train` in inclination_score/ and restart `make api` to score with the real model.'} Until then, scoring uses keyword rules in your browser, and only one conversation at a time.</PreviewNote></div>}

      {mode === 'single' ? (
        <div className="mx-auto grid max-w-5xl items-start gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,.8fr)]">
          <Card className="overflow-hidden">
            <CardHeader className="border-b border-border/70"><CardTitle>Add your conversation</CardTitle><p className="text-xs text-muted-foreground">Upload a text, CSV or JSON file—or paste messages below.</p></CardHeader>
            <CardContent className="p-4 sm:p-6">
              <Tabs value={inputTab} onValueChange={setInputTab}>
                <TabsList className="mb-4"><TabsTrigger value="paste">Paste text</TabsTrigger><TabsTrigger value="upload">Upload a file</TabsTrigger></TabsList>
                <TabsContent value="upload" className="mt-0">
                  <DropZone over={over} setOver={setOver} onFiles={(list) => loadFile(list[0])} onChoose={() => fileInput.current?.click()} hint=".txt, .csv or .json · one conversation at a time" />
                  <input ref={fileInput} type="file" accept=".txt,.csv,.json,text/plain,application/json" hidden onChange={(event) => { loadFile(event.target.files[0]); event.target.value = ''; }} />
                </TabsContent>
                <TabsContent value="paste" className="mt-0">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <label htmlFor="script" className="text-xs font-semibold text-muted-foreground">One message per line, as Speaker: message</label>
                    <Button type="button" variant="outline" size="sm" onClick={() => { setRaw(SAMPLE); setTitle('Support team, 40 seats'); setRoles({}); }}>Try a sample</Button>
                  </div>
                  <textarea id="script" value={raw} onChange={(event) => setRaw(event.target.value)}
                    placeholder={'Rep: Hi, thanks for joining.\nCustomer: Happy to. How much is the team plan?'}
                    className="min-h-64 w-full resize-y rounded-2xl border border-input bg-background p-4 text-sm leading-6 outline-none placeholder:text-muted-foreground/70 focus:ring-2 focus:ring-ring" />
                </TabsContent>
              </Tabs>
              {raw.trim() && turns.length === 0 && !error && <Warning>No messages found. Use one “Speaker: message” per line, or provide JSON turns / CSV columns for speaker and text.</Warning>}
              {error && <Alert>{error}</Alert>}
            </CardContent>
          </Card>
          <Card className="lg:sticky lg:top-28">
            <CardHeader><CardTitle>Before we score</CardTitle><p className="text-xs text-muted-foreground">Three quick steps.</p></CardHeader>
            <CardContent className="space-y-5">
              <label className="block space-y-2 text-xs font-semibold">1. Name this conversation<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Acme renewal call" aria-label="Conversation name" className="h-10 w-full rounded-xl border border-input bg-background px-3 text-sm font-normal outline-none focus:ring-2 focus:ring-ring" /></label>
              <div><p className="text-xs font-semibold">2. Mark each speaker</p>
                {speakers.length === 0 ? <p className="mt-1 text-xs leading-5 text-muted-foreground">Speakers will appear here once you add a conversation.</p> : <>
                  <SpeakerRoles rows={speakers.map((speaker) => ({ speaker, role: roleOf(speaker) }))} setRoles={setRoles} />
                  <p className="mt-2 text-[11px] text-muted-foreground">{turns.length} messages found.</p>
                  {!bothSides && <p className="mt-1 text-xs font-semibold text-rose-700 dark:text-rose-300">Mark at least one speaker as the customer and one as the sales side.</p>}
                </>}
              </div>
              {modelPicker}
              <Button type="button" className="w-full rounded-xl" disabled={turns.length < 2 || !bothSides || running || trained === null} onClick={run}>
                {running ? <><LoaderCircle size={16} className="animate-spin" />Reading the conversation…</> : <><Sparkles size={16} />Score conversation</>}
              </Button>
              {running && <Working label="Reading the conversation" />}
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="mx-auto grid max-w-5xl items-start gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,.8fr)]">
          <Card className="overflow-hidden">
            <CardHeader className="border-b border-border/70"><CardTitle>Add your conversations</CardTitle><p className="text-xs text-muted-foreground">Upload several files, one file with many conversations, or paste them below.</p></CardHeader>
            <CardContent className="space-y-4 p-4 sm:p-6">
              <DropZone over={over} setOver={setOver} onFiles={loadFiles} onChoose={() => fileInput.current?.click()} multiple
                hint="Several .txt, .csv or .json files, or one file with many conversations" />
              <input ref={fileInput} type="file" multiple accept=".txt,.csv,.json,text/plain,application/json" hidden onChange={(event) => { loadFiles(event.target.files); event.target.value = ''; }} />
              <details className="rounded-xl border border-border bg-muted/30 px-4 py-3 text-xs leading-5 text-muted-foreground">
                <summary className="cursor-pointer font-semibold text-foreground">Accepted formats</summary>
                <ul className="mt-2 list-disc space-y-1 pl-4">
                  <li><b>Text:</b> one “Speaker: message” per line. Separate conversations with a line of <code>---</code>; an optional <code># Title</code> line names each one.</li>
                  <li><b>CSV:</b> columns <code>speaker</code> (or <code>role</code>) and <code>text</code>, plus <code>conversation_id</code> to hold many conversations and an optional <code>title</code>.</li>
                  <li><b>JSON:</b> a list of <code>{'{ "title": …, "turns": [{ "speaker", "text" }] }'}</code>, or one conversation's list of turns.</li>
                  <li>Each file without its own titles is named after the file.</li>
                </ul>
              </details>
              {files.length > 0 && <ul className="space-y-2">{files.map((file) => (
                <li key={file.name} className="flex items-center justify-between gap-3 rounded-xl border border-border px-3 py-2 text-sm">
                  <span className="flex min-w-0 items-center gap-2"><Files size={15} className="shrink-0 text-primary" /><span className="truncate font-semibold">{file.name}</span></span>
                  <span className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground">{file.conversations.length} {file.conversations.length === 1 ? 'conversation' : 'conversations'}
                    <Button type="button" variant="ghost" size="icon" className="h-7 w-7" aria-label={`Remove ${file.name}`} onClick={() => setFiles((current) => current.filter((item) => item.name !== file.name))}><Trash2 size={14} /></Button></span>
                </li>
              ))}</ul>}
              <div>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <label htmlFor="batch-script" className="text-xs font-semibold text-muted-foreground">Or paste conversations, separated by a line of ---</label>
                  <Button type="button" variant="outline" size="sm" onClick={() => { setBatchText(SAMPLE_BATCH); setRoles({}); }}>Try a sample batch</Button>
                </div>
                <textarea id="batch-script" value={batchText} onChange={(event) => setBatchText(event.target.value)}
                  placeholder={'# Acme renewal\nRep: Hi, thanks for joining.\nCustomer: Happy to. How much is the team plan?\n---\n# Globex trial\nRep: How is the trial going?\nCustomer: Pretty well so far.'}
                  className="min-h-48 w-full resize-y rounded-2xl border border-input bg-background p-4 text-sm leading-6 outline-none placeholder:text-muted-foreground/70 focus:ring-2 focus:ring-ring" />
              </div>
              {batchText.trim() && parseTranscripts(batchText).length === 0 && <Warning>No messages found in the pasted text. Use one “Speaker: message” per line.</Warning>}
              {error && <Alert>{error}</Alert>}
            </CardContent>
          </Card>
          <Card className="lg:sticky lg:top-28">
            <CardHeader><CardTitle>Before we score</CardTitle><p className="text-xs text-muted-foreground">Check the speakers and the kind of conversation.</p></CardHeader>
            <CardContent className="space-y-5">
              <div><p className="text-xs font-semibold">1. Check who is who</p>
                {batchSpeakers.length === 0 ? <p className="mt-1 text-xs leading-5 text-muted-foreground">Speakers will appear here once you add conversations. Roles are guessed from names like “Rep” or “Customer”; change any that are wrong.</p> : <>
                  <div className="mt-2 max-h-64 overflow-y-auto pr-1"><SpeakerRoles rows={batchSpeakers} setRoles={setRoles} /></div>
                  <p className="mt-2 text-[11px] text-muted-foreground">{conversations.length} {conversations.length === 1 ? 'conversation' : 'conversations'} · {conversations.reduce((sum, c) => sum + c.turns.length, 0).toLocaleString()} messages</p>
                  {unusable.length > 0 && <p className="mt-1 text-xs font-semibold text-amber-700 dark:text-amber-300">{unusable.length} {unusable.length === 1 ? 'conversation needs' : 'conversations need'} a customer and a sales side, so {unusable.length === 1 ? 'it' : 'they'} will be skipped: {unusable.slice(0, 3).map((c) => c.title).join(', ')}{unusable.length > 3 ? '…' : ''}</p>}
                  {tooMany && <p className="mt-1 text-xs font-semibold text-rose-700 dark:text-rose-300">At most {MAX_CONVERSATIONS} conversations per batch; remove some files.</p>}
                </>}
              </div>
              {modelPicker}
              <Button type="button" className="w-full rounded-xl" disabled={batchCount === 0 || tooMany || running || !domains.length} onClick={runBatch}>
                {running ? <><LoaderCircle size={16} className="animate-spin" />Scoring {batchCount} conversations…</> : <><MessageSquareText size={16} />Score {batchCount || ''} {batchCount === 1 ? 'conversation' : 'conversations'}</>}
              </Button>
              {running && <Working label="Scoring every message and comparing sales moves" />}
            </CardContent>
          </Card>
        </div>
      )}
    </main>
  );
}

function DropZone({ over, setOver, onFiles, onChoose, hint, multiple = false }) {
  return (
    <div className={`flex min-h-52 flex-col items-center justify-center rounded-2xl border-2 border-dashed px-5 py-8 text-center transition-colors ${over ? 'border-primary bg-primary/5' : 'border-border bg-muted/35 hover:border-primary/40'}`}
      onDragOver={(event) => { event.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => { event.preventDefault(); setOver(false); onFiles(event.dataTransfer.files); }}>
      <span className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary"><FileUp size={25} /></span>
      <p className="font-bold">{multiple ? 'Drop conversation files here' : 'Drop a conversation here'}</p><p className="mt-1 text-xs text-muted-foreground">{hint}</p>
      <Button type="button" variant="outline" className="mt-5" onClick={onChoose}><Upload size={15} />{multiple ? 'Choose files' : 'Choose file'}</Button>
    </div>
  );
}

function SpeakerRoles({ rows, setRoles }) {
  return (
    <ul className="mt-2 space-y-2">{rows.map(({ speaker, role, chats }) => (
      <li key={speaker} className="flex items-center justify-between gap-2 rounded-xl border border-border px-3 py-2">
        <span className="min-w-0"><span className="block truncate text-sm font-semibold">{speaker}</span>{chats != null && <span className="text-[10px] text-muted-foreground">in {chats} {chats === 1 ? 'chat' : 'chats'}</span>}</span>
        <Select value={role} onValueChange={(value) => setRoles((current) => ({ ...current, [speaker]: value }))}><SelectTrigger aria-label={`Role of ${speaker}`} className="h-8 w-36 shrink-0 text-xs"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="agent">Sales side</SelectItem><SelectItem value="customer">Customer</SelectItem></SelectContent></Select>
      </li>
    ))}</ul>
  );
}

function Working({ label }) {
  return <div role="status" className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground"><span>{label}</span><span className="flex gap-1"><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary [animation-delay:-.2s]" /><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary [animation-delay:-.1s]" /><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary" /></span></div>;
}
