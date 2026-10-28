import { useMemo, useState } from 'react';
import { ArrowDownRight, ArrowLeft, ArrowUpRight, Columns3, Download, Flame, ListOrdered, Search, Table2, Target, Users, X } from 'lucide-react';
import ConversationDetail from './ConversationDetail';
import { ExampleCard, MoveColumn, positionLabel, splitEffects, SummaryCard } from './InsightViews';
import { LeadCard, ScoreCell, SortHead } from './LeadViews';
import { cap, EmptyIllustration, pct, Sparkline, StatusTag, useSelection } from './ui';
import { AGENT_MOVES, CUSTOMER_CUES, DATASETS, MODELS, pretty, STATUSES } from '../data/taxonomy';
import {
  Badge, Button, Card, CardContent, CardHeader, CardTitle, Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle,
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
  Tabs, TabsContent, TabsList, TabsTrigger, Toggle,
} from './ui/primitives';

const STATUS_ORDER = Object.fromEntries(STATUSES.map((status, index) => [status, index]));

function csvCell(value) {
  const text = value == null ? '' : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadLeads(leads, rank, model) {
  const header = ['rank', 'conversation', 'chance_to_buy', 'temperature', 'reason', 'peak_score', 'trend', 'turning_points', 'messages', 'model'];
  const rows = [...leads].sort((a, b) => rank[a.conversation_id] - rank[b.conversation_id]).map((lead) => [
    rank[lead.conversation_id], lead.title, lead.final_score.toFixed(4), lead.status, lead.status_reason,
    lead.max_score.toFixed(4), lead.trend.toFixed(4), lead.n_turning_points, lead.n_turns, model,
  ]);
  const body = [header, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([body], { type: 'text/csv;charset=utf-8' }));
  const link = Object.assign(document.createElement('a'), { href: url, download: 'lead-list.csv' });
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Lead list, script insights and cue counts for a batch of uploaded conversations. */
export default function BatchResults({ result, onEdit, onReset }) {
  const [tab, setTab] = useState('leads');
  const [openId, setOpenId] = useState(null);
  const leads = result.leads;
  const rank = useMemo(() => Object.fromEntries([...leads].sort((a, b) => b.final_score - a.final_score)
    .map((lead, index) => [lead.conversation_id, index + 1])), [leads]);
  const counts = Object.fromEntries(STATUSES.map((status) => [status, leads.filter((lead) => lead.status === status).length]));
  const average = leads.reduce((sum, lead) => sum + lead.final_score, 0) / leads.length;
  const top = leads.find((lead) => rank[lead.conversation_id] === 1);
  const dataset = DATASETS[result.dataset];
  const opened = openId ? result.conversations[openId] : null;

  return (
    <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
      <div className="mb-5 flex flex-wrap gap-2">
        <Button variant="outline" size="sm" onClick={onEdit}><ArrowLeft size={15} />Edit upload</Button>
        <Button variant="ghost" size="sm" onClick={onReset}>Analyze another batch</Button>
      </div>
      <header className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div>
          <p className="mb-2 text-xs font-bold uppercase tracking-[.16em] text-primary">Your conversations</p>
          <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Lead list and script analysis</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
            Scored by {MODELS[result.model]?.label ?? result.model}, trained on {dataset?.displayLabel ?? dataset?.label ?? result.dataset}.
            Models trained on one kind of conversation transfer poorly to others, so treat scores as a rough guide.
          </p>
        </div>
        <Button variant="outline" className="w-fit rounded-xl" onClick={() => downloadLeads(leads, rank, result.model)}><Download size={16} />Download lead list (CSV)</Button>
      </header>

      <section className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric icon={Users} label="Conversations scored" value={leads.length.toLocaleString()} secondary={result.skipped.length ? `${result.skipped.length} skipped` : `${result.n_turns.toLocaleString()} messages`} />
        <Metric icon={Flame} label="Hot leads" value={counts.Hot} secondary={`${counts.Rising} rising · ${counts.Stalled} stalled · ${counts.Cooling} cooling`} />
        <Metric icon={Target} label="Average chance to buy" value={pct(average)} />
        <Metric icon={ListOrdered} label="Top lead" value={top ? pct(top.final_score) : 'n/a'} secondary={top?.title} onClick={top ? () => setOpenId(top.conversation_id) : undefined} />
      </section>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="mb-5 flex-wrap">
          <TabsTrigger value="leads">Lead list</TabsTrigger>
          <TabsTrigger value="insights">Script insights</TabsTrigger>
          <TabsTrigger value="signals">Customer signals</TabsTrigger>
          {result.skipped.length > 0 && <TabsTrigger value="skipped">Skipped ({result.skipped.length})</TabsTrigger>}
        </TabsList>
        <TabsContent value="leads" className="mt-0"><LeadList leads={leads} rank={rank} onOpen={setOpenId} /></TabsContent>
        <TabsContent value="insights" className="mt-0"><BatchInsights result={result} onOpen={setOpenId} /></TabsContent>
        <TabsContent value="signals" className="mt-0"><Signals result={result} /></TabsContent>
        <TabsContent value="skipped" className="mt-0">
          <Card><CardContent className="space-y-2 p-4">
            <p className="text-sm text-muted-foreground">These conversations were not scored. Fix them and upload again.</p>
            {result.skipped.map((item, position) => <div key={position} className="rounded-xl border border-border p-3 text-sm"><b>{item.title}</b><p className="text-muted-foreground">{item.reason}</p></div>)}
          </CardContent></Card>
        </TabsContent>
      </Tabs>

      <Dialog open={Boolean(opened)} onOpenChange={(open) => !open && setOpenId(null)}>
        <DialogContent className="left-auto right-0 top-0 h-[100dvh] w-full max-w-[1100px] translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none border-l p-0 sm:rounded-l-3xl">
          <div className="sr-only"><DialogTitle>{opened?.title ?? 'Conversation'}</DialogTitle><DialogDescription>Score, turning points and transcript for this conversation.</DialogDescription></div>
          <DialogClose asChild><Button variant="outline" size="icon" aria-label="Close conversation" className="absolute right-4 top-4 z-30 rounded-full bg-card/90"><X size={17} /></Button></DialogClose>
          <div className="h-full overflow-y-auto">{opened && <ConversationDetail key={openId} convo={opened} model={result.model} split={null} />}</div>
        </DialogContent>
      </Dialog>
    </main>
  );
}

function Metric({ icon: Icon, label, value, secondary, onClick }) {
  const body = <CardContent className="flex items-center gap-3 p-4"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary"><Icon size={18} /></span><div className="min-w-0 text-left"><p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p><p className="truncate text-xl font-extrabold tabular-nums">{value}</p>{secondary && <p className="truncate text-[11px] text-muted-foreground" title={secondary}>{secondary}</p>}</div></CardContent>;
  return onClick
    ? <Card className="cursor-pointer transition-colors hover:border-primary/30"><button type="button" onClick={onClick} className="w-full rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">{body}</button></Card>
    : <Card>{body}</Card>;
}

function LeadList({ leads, rank, onOpen }) {
  const [view, setView] = useState('table');
  const [statuses, setStatuses] = useState([]);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ key: 'final_score', dir: -1 });
  const filtered = useMemo(() => {
    const search = query.trim().toLowerCase();
    return leads.filter((lead) => !statuses.length || statuses.includes(lead.status))
      .filter((lead) => !search || lead.title.toLowerCase().includes(search))
      .sort((a, b) => {
        const aValue = sort.key === 'status' ? STATUS_ORDER[a.status] : a[sort.key];
        const bValue = sort.key === 'status' ? STATUS_ORDER[b.status] : b[sort.key];
        return (aValue > bValue ? 1 : aValue < bValue ? -1 : 0) * sort.dir;
      });
  }, [leads, statuses, query, sort]);
  const sortBy = (key) => setSort((current) => ({ key, dir: current.key === key ? -current.dir : -1 }));
  const toggle = (status) => setStatuses((current) => (current.includes(status) ? current.filter((item) => item !== status) : [...current, status]));

  return (
    <>
      <section className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card px-4 py-3 shadow-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span className="relative"><Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search conversations" aria-label="Search conversation titles" className="h-9 w-56 rounded-xl border border-input bg-background pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring" /></span>
          {STATUSES.map((status) => <Toggle key={status} pressed={statuses.includes(status)} onPressedChange={() => toggle(status)} aria-label={`Show ${status} leads`} className="h-9 rounded-xl border border-transparent px-1.5 data-[state=on]:border-primary/40 data-[state=on]:bg-primary/10"><StatusTag status={status} /></Toggle>)}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold">{filtered.length} <span className="font-normal text-muted-foreground">of {leads.length}</span></span>
          <div className="inline-flex rounded-xl bg-muted p-1" role="group" aria-label="Lead layout">
            <Toggle pressed={view === 'table'} onPressedChange={() => setView('table')} aria-label="Table view" className="h-8 gap-2 rounded-lg px-3 text-xs data-[state=on]:bg-card data-[state=on]:shadow-sm"><Table2 size={15} /><span className="hidden sm:inline">Ranked</span></Toggle>
            <Toggle pressed={view === 'board'} onPressedChange={() => setView('board')} aria-label="Board view" className="h-8 gap-2 rounded-lg px-3 text-xs data-[state=on]:bg-card data-[state=on]:shadow-sm"><Columns3 size={15} /><span className="hidden sm:inline">Board</span></Toggle>
          </div>
        </div>
      </section>
      {filtered.length === 0 ? (
        <Card><CardContent className="flex flex-col items-center gap-3 py-14 text-center"><EmptyIllustration /><h2 className="text-lg font-bold">No conversations match</h2><Button variant="outline" onClick={() => { setStatuses([]); setQuery(''); }}>Clear filters</Button></CardContent></Card>
      ) : view === 'board' ? (
        <div className="grid min-w-0 grid-cols-1 items-start gap-4 sm:grid-cols-2 xl:grid-cols-[repeat(4,minmax(0,1fr))]">
          {STATUSES.map((status) => {
            const rows = filtered.filter((lead) => lead.status === status);
            return (
              <section key={status} className="max-h-[min(72vh,760px)] min-w-0 overflow-y-auto overscroll-contain rounded-2xl border border-border bg-muted/60 p-3.5">
                <div className="sticky top-0 z-10 -mx-3.5 -mt-3.5 mb-3 flex items-center justify-between border-b border-border/70 bg-muted/95 px-4 py-3 backdrop-blur"><StatusTag status={status} /><Badge variant="outline" className="bg-card tabular-nums">{rows.length}</Badge></div>
                <div className="grid gap-2.5">
                  {rows.map((lead) => <LeadCard key={lead.conversation_id} lead={lead} rank={rank[lead.conversation_id]} onClick={() => onOpen(lead.conversation_id)} />)}
                  {rows.length === 0 && <p className="rounded-xl border border-dashed border-border bg-card/60 px-3 py-6 text-center text-xs text-muted-foreground">No leads in this temperature.</p>}
                </div>
              </section>
            );
          })}
        </div>
      ) : (
        <Card className="overflow-hidden">
          <div className="max-h-[calc(100vh-320px)] min-h-72 overflow-auto">
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-14">Rank</TableHead>
                  <SortHead label="Conversation" onClick={() => sortBy('title')} />
                  <SortHead label="Chance to buy" onClick={() => sortBy('final_score')} />
                  <TableHead>Trend</TableHead>
                  <SortHead label="Temperature" onClick={() => sortBy('status')} />
                  <SortHead label="Reason" onClick={() => sortBy('status_reason')} />
                  <SortHead label="Turning points" onClick={() => sortBy('n_turning_points')} />
                  <SortHead label="Messages" onClick={() => sortBy('n_turns')} />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((lead) => (
                  <TableRow key={lead.conversation_id} tabIndex={0} role="button" onClick={() => onOpen(lead.conversation_id)}
                    onKeyDown={(event) => event.key === 'Enter' && onOpen(lead.conversation_id)}
                    className="cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring">
                    <TableCell className="tabular-nums font-bold text-muted-foreground">#{rank[lead.conversation_id]}</TableCell>
                    <TableCell className="max-w-64 font-semibold">{lead.title}</TableCell>
                    <TableCell><ScoreCell lead={lead} /></TableCell>
                    <TableCell><Sparkline values={lead.curve} width={86} height={26} /></TableCell>
                    <TableCell><StatusTag status={lead.status} /></TableCell>
                    <TableCell className="max-w-52 text-muted-foreground">{cap(lead.status_reason)}</TableCell>
                    <TableCell className="tabular-nums">{lead.n_turning_points}</TableCell>
                    <TableCell className="tabular-nums">{lead.n_turns}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
      )}
    </>
  );
}

function BatchInsights({ result, onOpen }) {
  const { server } = useSelection();
  const positions = ['all', ...(server.config?.insights?.position_buckets ?? ['early', 'middle', 'late'])];
  const [position, setPosition] = useState('all');
  const [selectedMove, setSelectedMove] = useState('');
  const view = result.insights.filter((row) => row.position === position).sort((a, b) => b.mean_delta - a.mean_delta);
  const { helpful, harmful, max } = splitEffects(view);
  const moves = view.map((row) => row.move);
  const move = moves.includes(selectedMove) ? selectedMove : moves[0];
  const examples = result.insight_examples.filter((example) => example.move === move && example.position === position);
  const titles = Object.fromEntries(result.leads.map((lead) => [lead.conversation_id, lead.title]));

  if (!result.insights.length) {
    return <Card><CardContent className="flex flex-col items-center gap-3 py-14 text-center"><EmptyIllustration /><h2 className="text-lg font-bold">Not enough conversations yet</h2><p className="max-w-lg text-sm text-muted-foreground">Script insights need each sales move to appear at least {result.insights_min_count} times before a customer reply. Upload more conversations to see which moves help or hurt.</p></CardContent></Card>;
  }
  return (
    <>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card p-3.5 shadow-sm">
        <div className="text-sm"><span className="font-semibold">Timing</span><span className="ml-2 text-muted-foreground">How each move worked at different points in your conversations.</span></div>
        <Tabs value={position} onValueChange={setPosition}><TabsList aria-label="Position in conversation">{positions.map((value) => <TabsTrigger key={value} value={value}>{positionLabel(value)}</TabsTrigger>)}</TabsList></Tabs>
      </div>
      <div className="mb-5 grid gap-3 md:grid-cols-2">
        <SummaryCard direction="up" lead={helpful[0]} label={helpful[0] ? `${pretty(helpful[0].move)} helps most` : 'No clear helpful move'} />
        <SummaryCard direction="down" lead={harmful[0]} label={harmful[0] ? `${pretty(harmful[0].move)} is linked to the biggest drop` : 'No clear harmful move'} />
      </div>
      <p className="mb-5 text-xs leading-5 text-muted-foreground">
        Based on your {result.leads.length} conversations. Moves seen fewer than {result.insights_min_count} times are not shown. With a small batch the confidence ranges are wide: a range that crosses zero means the effect is not clear yet. These are associations, not proven causes.
      </p>
      {view.length === 0 ? (
        <Card><CardContent className="py-12 text-center text-sm text-muted-foreground">No move occurs often enough at this position. Try the whole conversation.</CardContent></Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          <MoveColumn title="What helps" description="Moves followed by the customer's score rising." icon={ArrowUpRight} tone="text-teal-700 dark:text-teal-200" rows={helpful} max={max} positive />
          <MoveColumn title="What hurts" description="Moves followed by the customer's score dropping." icon={ArrowDownRight} tone="text-orange-700 dark:text-orange-200" rows={harmful} max={max} />
        </div>
      )}
      {moves.length > 0 && (
        <Card className="mt-5">
          <CardHeader className="flex-row flex-wrap items-center justify-between gap-3 border-b border-border/70 pb-4">
            <div><CardTitle>Examples from your conversations</CardTitle><p className="mt-1 text-xs text-muted-foreground">The sales message and the customer's next reply.</p></div>
            <Select value={move ?? ''} onValueChange={setSelectedMove}><SelectTrigger className="w-56" aria-label="Sales move"><SelectValue placeholder="Choose a move" /></SelectTrigger><SelectContent>{moves.map((key) => <SelectItem key={key} value={key}>{pretty(key)}</SelectItem>)}</SelectContent></Select>
          </CardHeader>
          <CardContent className="grid gap-3 p-4">
            {examples.map((example) => <ExampleCard key={`${example.conversation_id}-${example.agent_turn}`} example={example}
              action={<button type="button" onClick={() => onOpen(example.conversation_id)} className="text-xs font-bold text-primary hover:underline">{titles[example.conversation_id]} →</button>} />)}
          </CardContent>
        </Card>
      )}
    </>
  );
}

function Signals({ result }) {
  const total = result.leads.length;
  const customer = result.cue_summary.filter((row) => row.kind === 'customer');
  const agent = result.cue_summary.filter((row) => row.kind === 'agent');
  const Row = ({ row, color, phrase }) => (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm"><span><b>{pretty(row.cue)}</b> <span className="text-xs text-muted-foreground">{phrase}</span></span><span className="shrink-0 tabular-nums text-xs text-muted-foreground">{row.n_conversations} of {total} chats · {row.n_turns} times</span></div>
      <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full" style={{ width: `${(row.n_conversations / total) * 100}%`, backgroundColor: color }} /></div>
    </div>
  );
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <Card>
        <CardHeader className="border-b border-border/70 pb-4"><CardTitle>What customers said</CardTitle><p className="text-xs text-muted-foreground">How many conversations show each buying signal or objection. Teal raises the chance to buy, orange lowers it.</p></CardHeader>
        <CardContent className="space-y-4 p-4">
          {customer.length === 0 && <p className="text-sm text-muted-foreground">No customer cues were detected.</p>}
          {customer.map((row) => <Row key={row.cue} row={row} phrase={CUSTOMER_CUES[row.cue]?.phrase} color={CUSTOMER_CUES[row.cue]?.direction > 0 ? '#0f9f88' : '#f97316'} />)}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="border-b border-border/70 pb-4"><CardTitle>What the sales side did</CardTitle><p className="text-xs text-muted-foreground">How many conversations use each sales move. See Script insights for how customers reacted.</p></CardHeader>
        <CardContent className="space-y-4 p-4">
          {agent.length === 0 && <p className="text-sm text-muted-foreground">No sales moves were detected.</p>}
          {agent.map((row) => <Row key={row.cue} row={row} phrase={AGENT_MOVES[row.cue]?.phrase} color="#2563eb" />)}
        </CardContent>
      </Card>
    </div>
  );
}
