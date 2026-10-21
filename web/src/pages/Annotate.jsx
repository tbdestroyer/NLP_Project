import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowLeft, ArrowRight, Save, Tag, UserRound } from 'lucide-react';
import { getAnnotations, saveAnnotations } from '../api/client';
import { DATASETS } from '../data/taxonomy';
import { ErrorNote, useAsync, useSelection } from '../components/ui';
import { Badge, Button, Card, CardContent, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Skeleton } from '../components/ui/primitives';

const DIRECTION_LABELS = { more: 'More inclined', same: 'Same', less: 'Less inclined' };

function safeName(value) {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
}

function ChatTurn({ turn, direction, turningPoint, directions, onChange, active, onFocus }) {
  return (
    <article onFocus={onFocus} onClick={onFocus} className={`grid gap-4 rounded-2xl border p-4 transition-colors md:grid-cols-[minmax(0,1fr)_260px] md:items-center ${active ? 'border-primary/40 bg-primary/[.035] ring-1 ring-primary/15' : 'border-border bg-card'}`}>
      <div className="min-w-0">
        <div className="mb-2 flex items-center gap-2"><Badge variant="outline" className={turn.role === 'customer' ? 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-900 dark:bg-blue-950/50 dark:text-blue-100' : 'bg-muted'}><UserRound size={11} />{turn.role === 'customer' ? 'Customer' : 'Agent'}</Badge><span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Turn {turn.turn}</span></div>
        <p className="text-sm leading-6">{turn.text}</p>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/70 pt-3 md:border-l md:border-t-0 md:pl-4 md:pt-0">
        {turn.role === 'customer' ? (
          <fieldset className="min-w-0 flex-1">
            <legend className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Inclination shift</legend>
            <div className="flex flex-wrap gap-1.5">
              {directions.map((option, index) => (
                <label key={option} className={`cursor-pointer rounded-lg border px-2.5 py-1.5 text-[11px] font-semibold transition-colors ${direction === option ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-background hover:bg-muted'}`}>
                  <input className="sr-only" type="radio" name={`direction-${turn.turn}`} value={option} checked={direction === option}
                    onChange={() => onChange({ direction: option })} aria-label={`${DIRECTION_LABELS[option] ?? option} (shortcut ${index + 1})`} />
                  {DIRECTION_LABELS[option] ?? option}
                </label>
              ))}
            </div>
          </fieldset>
        ) : <span className="text-[10px] text-muted-foreground">Agent turn</span>}
        <label className={`inline-flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-xs font-semibold ${turningPoint ? 'bg-amber-100 text-amber-900 dark:bg-amber-950/50 dark:text-amber-100' : 'bg-muted text-muted-foreground'}`}>
          <input type="checkbox" checked={turningPoint} onChange={(event) => onChange({ turning_point: event.target.checked })} className="h-4 w-4 accent-amber-600" />
          <Tag size={13} />Turning point
        </label>
      </div>
    </article>
  );
}

export default function Annotate() {
  const selection = useSelection();
  const directions = selection.server.config?.annotation?.directions ?? ['more', 'same', 'less'];
  const datasetOptions = Object.keys(selection.server.options).filter((key) => key in DATASETS);
  const [annotator, setAnnotator] = useState('');
  const [activeAnnotator, setActiveAnnotator] = useState('');
  const [scope, setScope] = useState('all');
  const [position, setPosition] = useState(0);
  const [activeTurn, setActiveTurn] = useState(null);
  const [draft, setDraft] = useState({});
  const [refresh, setRefresh] = useState(0);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [savedMessage, setSavedMessage] = useState('');

  const queueState = useAsync(
    () => activeAnnotator ? getAnnotations(activeAnnotator, scope) : null,
    [activeAnnotator, scope, refresh],
  );
  const conversations = queueState.data?.conversations ?? [];
  const current = conversations[position] ?? null;
  useEffect(() => {
    setActiveTurn(current?.turns[0]?.turn ?? null);
  }, [current?.conversation_id]);
  useEffect(() => {
    if (!current) {
      setDraft({});
      return;
    }
    const currentLabels = Object.fromEntries(current.labels.map((label) => [label.turn, label]));
    setDraft(Object.fromEntries(current.turns.map((turn) => {
      const previous = currentLabels[turn.turn];
      return [turn.turn, {
        direction: previous?.direction ?? null,
        turning_point: Boolean(previous?.turning_point),
      }];
    })));
    setSaveError('');
    setSavedMessage('');
  }, [current?.conversation_id, current?.labels]);

  useEffect(() => {
    if (!conversations.length) {
      setPosition(0);
      return;
    }
    const firstUnlabelled = conversations.findIndex((conversation) => !conversation.labelled);
    setPosition(firstUnlabelled < 0 ? 0 : firstUnlabelled);
  }, [activeAnnotator, scope, queueState.data]);

  const updateTurn = (turn, patch) => {
    setDraft((currentDraft) => ({
      ...currentDraft,
      [turn]: { ...currentDraft[turn], ...patch },
    }));
    setSaveError('');
    setSavedMessage('');
  };
  const goTo = (index) => setPosition(Math.max(0, Math.min(index, conversations.length - 1)));
  const nextUnlabelled = () => {
    if (!conversations.length) return;
    const order = [...Array(conversations.length).keys()].slice(position + 1)
      .concat([...Array(conversations.length).keys()].slice(0, position + 1));
    goTo(order.find((index) => !conversations[index].labelled) ?? position);
  };
  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      const textEntry = target instanceof HTMLElement && (
        target.isContentEditable || ['SELECT', 'TEXTAREA'].includes(target.tagName)
        || (target instanceof HTMLInputElement && ['text', 'search', 'number'].includes(target.type))
      );
      if (textEntry) return;
      const radioFocused = target instanceof HTMLInputElement && target.type === 'radio';
      if (event.key === 'ArrowLeft' && position > 0 && !radioFocused) { event.preventDefault(); goTo(position - 1); }
      else if (event.key === 'ArrowRight' && position < conversations.length - 1 && !radioFocused) { event.preventDefault(); goTo(position + 1); }
      else if (/^[123]$/.test(event.key) && activeTurn != null && current) {
        const turn = current.turns.find((item) => item.turn === activeTurn);
        const direction = directions[Number(event.key) - 1];
        if (turn?.role === 'customer' && direction) updateTurn(activeTurn, { direction });
      } else if (event.key.toLowerCase() === 't' && activeTurn != null && current) {
        const turn = current.turns.find((item) => item.turn === activeTurn);
        if (turn) updateTurn(activeTurn, { turning_point: !draft[activeTurn]?.turning_point });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [position, conversations.length, activeTurn, current, directions, draft]);
  const submit = async (event) => {
    event.preventDefault();
    if (!current || !activeAnnotator) return;
    const missing = current.turns
      .filter((turn) => turn.role === 'customer' && !draft[turn.turn]?.direction)
      .map((turn) => turn.turn);
    if (missing.length) {
      setSaveError(`Choose more / same / less for every customer turn. Missing turns: ${missing.join(', ')}`);
      return;
    }
    setSaving(true);
    setSaveError('');
    setSavedMessage('');
    try {
      await saveAnnotations({
        annotator: activeAnnotator,
        conversation_id: current.conversation_id,
        dataset: current.dataset,
        labels: current.turns.map((turn) => ({
          turn: turn.turn,
          role: turn.role,
          direction: turn.role === 'customer' ? draft[turn.turn]?.direction ?? null : null,
          turning_point: Boolean(draft[turn.turn]?.turning_point),
        })),
      });
      setSavedMessage('Saved.');
      setRefresh((value) => value + 1);
    } catch (error) {
      setSaveError(error.message);
    } finally {
      setSaving(false);
    }
  };

  const completed = queueState.data?.completed ?? 0;
  const total = queueState.data?.total ?? 0;
  return (
    <main className="min-h-[calc(100vh-72px)] bg-gradient-to-b from-primary/[.035] via-background to-background px-4 py-7 md:px-8 md:py-10">
      <div className="mx-auto max-w-[1320px]">
        <header className="mb-6 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div><div className="mb-2 flex items-center gap-2"><Badge variant="outline" className="border-primary/20 bg-primary/5 text-primary">Team tool</Badge><Badge variant="secondary">Internal</Badge></div><h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Conversation labelling</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">Label how the customer's inclination changes, turn by turn. Model scores are hidden here on purpose.</p></div>
          <div className="flex flex-wrap gap-2 text-[10px] font-medium text-muted-foreground"><kbd className="rounded-md border border-border bg-card px-1.5 py-1">1 / 2 / 3</kbd> choose inclination <kbd className="rounded-md border border-border bg-card px-1.5 py-1">T</kbd> turning point <kbd className="rounded-md border border-border bg-card px-1.5 py-1">← / →</kbd> navigate</div>
        </header>

        <Card className="mb-5">
          <CardContent className="flex flex-col gap-4 p-4 md:flex-row md:items-end">
            <label className="w-full space-y-1.5 text-xs font-semibold md:max-w-xs">Your annotator name<input value={annotator} onChange={(event) => setAnnotator(event.target.value)} placeholder="e.g. shantanu" className="h-10 w-full rounded-xl border border-input bg-background px-3 text-sm font-normal outline-none focus:ring-2 focus:ring-ring" /></label>
            <label className="w-full space-y-1.5 text-xs font-semibold md:max-w-xs">Dataset<Select value={scope} onValueChange={setScope}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All datasets</SelectItem>{datasetOptions.map((dataset) => <SelectItem key={dataset} value={dataset}>{DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label ?? dataset}</SelectItem>)}</SelectContent></Select></label>
            <Button type="button" disabled={!safeName(annotator)} onClick={() => { setActiveAnnotator(safeName(annotator)); setRefresh((value) => value + 1); }} className="shrink-0"><UserRound size={16} />Load annotation queue</Button>
            {activeAnnotator && <span className="text-[10px] leading-4 text-muted-foreground">Labels save to annotation/labels/{activeAnnotator}.jsonl. Use the same name each time.</span>}
          </CardContent>
        </Card>

        {queueState.error && <ErrorNote error={queueState.error} />}
        {activeAnnotator && queueState.loading && <div className="space-y-4"><Skeleton className="h-16" /><Skeleton className="h-80" /></div>}
        {activeAnnotator && queueState.data && (
          <>
            <Card className="mb-5">
              <CardContent className="flex flex-wrap items-center justify-between gap-4 p-4">
                <div className="min-w-48 flex-1"><div className="mb-2 flex justify-between text-xs font-semibold"><span>Labelling progress</span><span className="tabular-nums text-muted-foreground">{completed} of {total} labelled</span></div><div role="progressbar" aria-valuenow={completed} aria-valuemin={0} aria-valuemax={Math.max(total, 1)} className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-gradient-to-r from-blue-700 to-sky-400 transition-all" style={{ width: `${total ? completed / total * 100 : 0}%` }} /></div></div>
                {current && <div className="flex flex-wrap gap-2"><Badge variant="outline">Conversation {position + 1} of {conversations.length}</Badge><Badge variant="outline">{DATASETS[current.dataset]?.displayLabel ?? DATASETS[current.dataset]?.label ?? current.dataset}</Badge><Badge className={current.labelled ? 'bg-emerald-600' : 'bg-amber-600'}>{current.labelled ? 'Labelled' : 'Not labelled yet'}</Badge></div>}
              </CardContent>
            </Card>
            {!current ? <Card><CardContent className="py-16 text-center"><h2 className="text-lg font-bold">No conversations in this sample</h2></CardContent></Card> : (
              <>
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex gap-2"><Button variant="outline" disabled={position === 0} onClick={() => goTo(position - 1)}><ArrowLeft size={15} />Previous</Button><Button variant="outline" disabled={position === conversations.length - 1} onClick={() => goTo(position + 1)}>Next<ArrowRight size={15} /></Button></div>
                  <Button variant="ghost" size="sm" onClick={nextUnlabelled}>Next unlabelled</Button>
                </div>
                <details className="mb-4 rounded-2xl border border-border bg-card p-4">
                  <summary className="cursor-pointer text-sm font-bold">How to label</summary>
                  <ul className="mt-3 list-inside list-disc space-y-2 text-xs leading-5 text-muted-foreground">
                    <li>For every customer turn, decide whether the customer is more, the same or less inclined to buy than just before that turn.</li>
                    <li>Mark turning point on any turn, customer or agent, where the conversation clearly changed direction. Most conversations have zero to two.</li>
                    <li>Judge only from what has been said so far. Do not read ahead before deciding.</li>
                    <li>Full instructions and worked examples: <code>annotation/guidelines.md</code>.</li>
                  </ul>
                </details>
                <form className="space-y-3" onSubmit={submit}>
                  <AnimatePresence mode="popLayout">
                    {current.turns.map((turn) => <motion.div key={turn.turn} layout initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.18 }}>
                      <ChatTurn turn={turn} direction={draft[turn.turn]?.direction} turningPoint={Boolean(draft[turn.turn]?.turning_point)}
                        directions={directions} active={activeTurn === turn.turn} onFocus={() => setActiveTurn(turn.turn)}
                        onChange={(patch) => updateTurn(turn.turn, patch)} />
                    </motion.div>)}
                  </AnimatePresence>
                  {saveError && <div role="alert" className="rounded-xl border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100">{saveError}</div>}
                  {savedMessage && <div role="status" className="rounded-xl border border-emerald-300 bg-emerald-50 p-3 text-sm text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-100">{savedMessage}</div>}
                  <div className="sticky bottom-20 flex justify-end rounded-2xl border border-border bg-background/95 p-3 shadow-float backdrop-blur md:bottom-4"><Button type="submit" disabled={saving} className="min-h-11"><Save size={16} />{saving ? 'Saving…' : 'Save and go to the next unlabelled'}</Button></div>
                </form>
              </>
            )}
          </>
        )}
      </div>
    </main>
  );
}
