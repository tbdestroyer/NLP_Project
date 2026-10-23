import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, BadgeCheck, FileUp, LoaderCircle, Sparkles, Upload } from 'lucide-react';
import { analyzeConversation, getAnalyzeModels } from '../api/client';
import ConversationDetail from '../components/ConversationDetail';
import { Segmented } from '../components/ui';
import { DATASETS, MODELS } from '../data/taxonomy';
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/primitives';

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

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"' && quoted && text[i + 1] === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === ',' && !quoted) {
      row.push(field);
      field = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }
  row.push(field);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

/** Accepts "Speaker: text" lines, JSON turns, or CSV with speaker/role and text columns. */
function parse(raw) {
  const text = raw.trim();
  if (!text) return [];
  if (text.startsWith('[') || text.startsWith('{')) {
    try {
      const decoded = JSON.parse(text);
      const records = Array.isArray(decoded) ? decoded : decoded.turns;
      if (!Array.isArray(records)) return [];
      return records.map((t) => ({
        speaker: String(t.speaker ?? t.role ?? 'Unknown'),
        role: ['agent', 'customer'].includes(String(t.role).toLowerCase()) ? String(t.role).toLowerCase() : undefined,
        text: String(t.text ?? t.message ?? ''),
      }))
        .filter((t) => t.text.trim());
    } catch {
      return [];
    }
  }
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (/^\s*"?((speaker|role).*,.*(text|message)|(text|message).*,.*(speaker|role))"?\s*$/i.test(lines[0])) {
    const rows = parseCsv(text);
    const headers = rows.shift().map((header) => header.trim().toLowerCase());
    const speakerColumn = headers.findIndex((header) => ['speaker', 'role'].includes(header));
    const textColumn = headers.findIndex((header) => ['text', 'message'].includes(header));
    return rows.map((row) => {
      const roleOrSpeaker = row[speakerColumn]?.trim() || 'Unknown';
      return {
        speaker: row[headers.indexOf('speaker')]?.trim() || roleOrSpeaker,
        role: ['agent', 'customer'].includes(roleOrSpeaker.toLowerCase()) ? roleOrSpeaker.toLowerCase() : undefined,
        text: row[textColumn]?.trim() || '',
      };
    }).filter((turn) => turn.text);
  }
  const turns = [];
  for (const line of lines) {
    const m = line.match(/^\s*([^:]{1,30}):\s*(.+)$/);
    if (m) turns.push({ speaker: m[1].trim(), text: m[2].trim() });
    else if (turns.length) turns[turns.length - 1].text += ` ${line.trim()}`;
  }
  return turns;
}

const AGENT_HINT = /^(rep|agent|seller|sales|persuader|bot|assistant|support)/i;

// Each trained model learned from one kind of conversation; the user picks the closest one.
const DOMAINS = {
  saas: 'Sales call or chat',
  craigslist: 'Price negotiation',
  p4g: 'Persuasion or donation ask',
};
const PREFERRED_DOMAIN = 'saas';

const PreviewNote = ({ children }) => <div className="mb-5 flex items-start gap-3 rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-950 dark:border-blue-900 dark:bg-blue-950/30 dark:text-blue-100"><BadgeCheck size={17} className="mt-0.5 shrink-0" /><span>{children}</span></div>;

export default function Analyze() {
  const [trained, setTrained] = useState(null); // { [dataset]: [models] } from the server
  const [trainedError, setTrainedError] = useState('');
  const [dataset, setDataset] = useState(PREFERRED_DOMAIN);
  const domains = Object.keys(DOMAINS).filter((key) => trained?.[key]?.length);
  const preview = trained !== null && domains.length === 0;
  const availableModels = (trained?.[dataset] ?? []).filter((key) => key in MODELS);
  const [raw, setRaw] = useState('');
  const [title, setTitle] = useState('');
  const [inputTab, setInputTab] = useState('paste');
  const [model, setModel] = useState('v1');
  const [roles, setRoles] = useState({}); // speaker -> 'agent' | 'customer'
  const [over, setOver] = useState(false);
  const [status, setStatus] = useState('idle'); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInput = useRef(null);

  const turns = useMemo(() => parse(raw), [raw]);
  const speakers = useMemo(() => [...new Set(turns.map((t) => t.speaker))], [turns]);
  const declaredRoles = Object.fromEntries(turns.filter((t) => t.role).map((t) => [t.speaker, t.role]));
  const roleOf = (speaker) => roles[speaker] ?? declaredRoles[speaker]
    ?? (AGENT_HINT.test(speaker) ? 'agent' : speakers.find((sp) => AGENT_HINT.test(sp)) ? 'customer' : speaker === speakers[0] ? 'agent' : 'customer');
  const hasBothSides = speakers.some((sp) => roleOf(sp) === 'agent') && speakers.some((sp) => roleOf(sp) === 'customer');

  useEffect(() => {
    getAnalyzeModels().then(setTrained).catch((e) => { setTrainedError(e.message); setTrained({}); });
  }, []);
  useEffect(() => {
    if (domains.length && !domains.includes(dataset)) setDataset(domains[0]);
  }, [domains.join(','), dataset]);
  useEffect(() => {
    if (availableModels.length && !availableModels.includes(model)) setModel(availableModels[0]);
  }, [availableModels.join(','), model]);

  const loadFile = async (file) => {
    if (!file) return;
    const extension = file.name.toLowerCase().split('.').pop();
    if (!['txt', 'csv', 'json'].includes(extension)) {
      setError('Choose a .txt, .csv or .json conversation file.');
      return;
    }
    if (file.size > 2_000_000) { setError('That file is over 2 MB. Upload a single conversation.'); return; }
    setError('');
    try {
      setRaw(await file.text());
    } catch (readError) {
      setError(`Could not read that file: ${readError.message}`);
      return;
    }
    setInputTab('paste');
    if (!title) setTitle(file.name.replace(/\.[^.]+$/, ''));
  };

  const run = async () => {
    setStatus('running');
    setError('');
    try {
      const out = await analyzeConversation({
        title, dataset, model, preview, turns: turns.map((t) => ({ role: roleOf(t.speaker), text: t.text, speaker: t.speaker })),
      });
      setResult(out);
      setStatus('done');
    } catch (e) {
      setError(e.message);
      setStatus('error');
    }
  };

  if (status === 'done' && result) {
    return (
      <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
        <div className="mb-5 flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => setStatus('idle')}><ArrowLeft size={15} />Edit conversation</Button>
          <Button variant="ghost" size="sm" onClick={() => { setRaw(''); setTitle(''); setRoles({}); setResult(null); setStatus('idle'); setInputTab('paste'); }}>Analyze another</Button>
        </div>
        {result.model === 'preview'
          ? <PreviewNote><b>Preview: no trained model was available.</b><br />These numbers come from simple keyword rules in the browser, not from the team's models.</PreviewNote>
          : <PreviewNote><b>Scored by the {MODELS[result.model]?.label ?? result.model} model trained on {DATASETS[result.dataset]?.displayLabel ?? DATASETS[result.dataset]?.label ?? result.dataset}.</b><br />Models trained on one kind of conversation transfer poorly to others, so treat the score as a rough guide.</PreviewNote>}
        <ConversationDetail convo={result} model={result.model} />
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[1080px] px-5 py-8 md:px-8 md:py-12">
      <header className="mx-auto mb-8 max-w-2xl text-center">
        <Badge variant="outline" className="mb-4 border-primary/20 bg-primary/5 text-primary"><Sparkles size={13} />Analyze a conversation</Badge>
        <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Read the temperature of a new conversation</h1>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-muted-foreground">Add a conversation to see its score after each message, likely turning points, and what may have changed the customer's mind.</p>
      </header>
      {preview && <div className="mx-auto max-w-3xl"><PreviewNote><b>Preview: no trained model found.</b><br />{trainedError ? `The data server did not answer (${trainedError}). Start it with \`make api\`.` : 'Run `make train` in inclination_score/ and restart `make api` to score with the real model.'} Until then, scoring uses keyword rules in your browser.</PreviewNote></div>}
      <div className="mx-auto grid max-w-5xl items-start gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,.8fr)]">
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-border/70"><CardTitle>Add your conversation</CardTitle><p className="text-xs text-muted-foreground">Upload a text, CSV or JSON file—or paste messages below.</p></CardHeader>
          <CardContent className="p-4 sm:p-6">
            <Tabs value={inputTab} onValueChange={setInputTab}>
              <TabsList className="mb-4"><TabsTrigger value="paste">Paste text</TabsTrigger><TabsTrigger value="upload">Upload a file</TabsTrigger></TabsList>
              <TabsContent value="upload" className="mt-0">
                <div className={`flex min-h-60 flex-col items-center justify-center rounded-2xl border-2 border-dashed px-5 py-9 text-center transition-colors ${over ? 'border-primary bg-primary/5' : 'border-border bg-muted/35 hover:border-primary/40'}`}
                  onDragOver={(event) => { event.preventDefault(); setOver(true); }}
                  onDragLeave={() => setOver(false)}
                  onDrop={(event) => { event.preventDefault(); setOver(false); loadFile(event.dataTransfer.files[0]); }}>
                  <span className="mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary"><FileUp size={25} /></span>
                  <p className="font-bold">Drop a conversation here</p><p className="mt-1 text-xs text-muted-foreground">.txt, .csv or .json · one conversation at a time</p>
                  <Button type="button" variant="outline" className="mt-5" onClick={() => fileInput.current?.click()}><Upload size={15} />Choose file</Button>
                  <input ref={fileInput} type="file" accept=".txt,.csv,.json,text/plain,application/json" hidden onChange={(event) => loadFile(event.target.files[0])} />
                </div>
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
            {raw.trim() && turns.length === 0 && !error && <p role="alert" className="mt-3 rounded-xl border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100">No messages found. Use one “Speaker: message” per line, or provide JSON turns / CSV columns for speaker and text.</p>}
            {error && <div role="alert" className="mt-3 rounded-xl border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">{error}</div>}
          </CardContent>
        </Card>
        <Card className="lg:sticky lg:top-28">
          <CardHeader><CardTitle>Before we score</CardTitle><p className="text-xs text-muted-foreground">Three quick steps to make the preview useful.</p></CardHeader>
          <CardContent className="space-y-5">
            <label className="block space-y-2 text-xs font-semibold">1. Name this conversation<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Acme renewal call" aria-label="Conversation name" className="h-10 w-full rounded-xl border border-input bg-background px-3 text-sm font-normal outline-none focus:ring-2 focus:ring-ring" /></label>
            <div><p className="text-xs font-semibold">2. Mark each speaker</p>
              {speakers.length === 0 ? <p className="mt-1 text-xs leading-5 text-muted-foreground">Speakers will appear here once you add a conversation.</p> : <>
                <ul className="mt-2 space-y-2">{speakers.map((speaker) => <li key={speaker} className="flex items-center justify-between gap-2 rounded-xl border border-border px-3 py-2"><span className="truncate text-sm font-semibold">{speaker}</span><Select value={roleOf(speaker)} onValueChange={(role) => setRoles((current) => ({ ...current, [speaker]: role }))}><SelectTrigger aria-label={`Role of ${speaker}`} className="h-8 w-36 text-xs"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="agent">Sales side</SelectItem><SelectItem value="customer">Customer</SelectItem></SelectContent></Select></li>)}</ul>
                <p className="mt-2 text-[11px] text-muted-foreground">{turns.length} messages found.</p>
                {!hasBothSides && <p className="mt-1 text-xs font-semibold text-rose-700 dark:text-rose-300">Mark at least one speaker as the customer and one as the sales side.</p>}
              </>}
            </div>
            {domains.length > 0 && <div><p className="mb-2 text-xs font-semibold">3. What kind of conversation is it?</p>
              <Select value={dataset} onValueChange={setDataset}><SelectTrigger aria-label="Conversation type" className="h-10 w-full text-sm"><SelectValue /></SelectTrigger><SelectContent>{domains.map((key) => <SelectItem key={key} value={key}>{DOMAINS[key]}</SelectItem>)}</SelectContent></Select>
              <p className="mt-1 text-[11px] text-muted-foreground">Uses the model trained on {DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label}.</p>
              {availableModels.length > 1 && <div className="mt-3"><Segmented label="Model" value={model} onChange={setModel} options={availableModels.map((key) => ({ value: key, label: MODELS[key].label }))} /></div>}
            </div>}
            <Button type="button" className="w-full rounded-xl" disabled={turns.length < 2 || !hasBothSides || status === 'running' || trained === null} onClick={run}>
              {status === 'running' ? <><LoaderCircle size={16} className="animate-spin" />Reading the conversation…</> : <><Sparkles size={16} />Score conversation</>}
            </Button>
            {status === 'running' && <div role="status" className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground"><span>Reading the conversation</span><span className="flex gap-1"><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary [animation-delay:-.2s]" /><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary [animation-delay:-.1s]" /><i className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary" /></span></div>}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
