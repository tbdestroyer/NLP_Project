import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowDownUp, Columns3, Filter, ListFilter, Search, SlidersHorizontal, Table2, X } from 'lucide-react';
import { getLeads, searchMessages } from '../api/client';
import { cap, EmptyIllustration, ErrorNote, pct, Sparkline, StatusTag, SyntheticNote, useAsync, useSelection } from '../components/ui';
import { STATUSES } from '../data/taxonomy';
import {
  Badge, Button, Card, CardContent, Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle,
  Slider, Table, TableBody, TableCell, TableHead, TableHeader, TableRow, Toggle, Tooltip, TooltipContent, TooltipTrigger,
} from '../components/ui/primitives';

const STATUS_ORDER = Object.fromEntries(STATUSES.map((status, index) => [status, index]));

export default function Leads() {
  const selection = useSelection();
  const navigate = useNavigate();
  const location = useLocation();
  const { data: leads, error } = useAsync(() => getLeads(selection), [selection.dataset, selection.model, selection.split]);
  const [statuses, setStatuses] = useState([]);
  const [query, setQuery] = useState('');
  const [scoreRange, setScoreRange] = useState([0, 100]);
  const [minTurns, setMinTurns] = useState(1);
  const [sort, setSort] = useState({ key: 'final_score', dir: -1 });
  const [mentioned, setMentioned] = useState(null);
  const [searchError, setSearchError] = useState('');
  const [filterOpen, setFilterOpen] = useState(false);
  const [view, setView] = useState('board');

  useEffect(() => {
    const search = query.trim();
    if (!search) { setMentioned(null); setSearchError(''); return undefined; }
    let alive = true;
    setSearchError('');
    const timer = setTimeout(() => searchMessages(selection.dataset, search)
      .then((ids) => alive && setMentioned(ids))
      .catch((cause) => {
        if (alive) {
          setMentioned(new Set());
          setSearchError(`Message search failed: ${cause.message}`);
        }
      }), 200);
    return () => { alive = false; clearTimeout(timer); };
  }, [query, selection.dataset]);

  const filtered = useMemo(() => {
    if (!leads) return [];
    const search = query.trim().toLowerCase();
    return leads.filter((lead) => !statuses.length || statuses.includes(lead.status))
      .filter((lead) => lead.final_score >= scoreRange[0] / 100 && lead.final_score <= scoreRange[1] / 100)
      .filter((lead) => lead.n_turns >= minTurns)
      .filter((lead) => !search || lead.title.toLowerCase().includes(search) || mentioned?.has(lead.conversation_id))
      .sort((a, b) => {
        const aValue = sort.key === 'status' ? STATUS_ORDER[a.status] : a[sort.key];
        const bValue = sort.key === 'status' ? STATUS_ORDER[b.status] : b[sort.key];
        return (aValue > bValue ? 1 : aValue < bValue ? -1 : 0) * sort.dir;
      });
  }, [leads, statuses, query, scoreRange, minTurns, mentioned, sort]);

  if (error) return <ErrorNote error={error} />;
  if (!leads) return <LeadsSkeleton />;

  const topKCount = Math.max(...(selection.server.config?.eval?.precision_at ?? [20]));
  const topK = filtered.slice(0, topKCount);
  const baseRate = leads.reduce((sum, lead) => sum + lead.outcome, 0) / (leads.length || 1);
  const activeFilterCount = Number(statuses.length > 0) + Number(scoreRange[0] > 0 || scoreRange[1] < 100) + Number(minTurns > 1) + Number(Boolean(query.trim()));

  const openConversation = (lead) => navigate(`/conversations/${encodeURIComponent(lead.conversation_id)}`, {
    state: { backgroundLocation: location },
  });
  const sortBy = (key) => setSort((current) => ({ key, dir: current.key === key ? -current.dir : -1 }));
  const clearFilters = () => { setStatuses([]); setQuery(''); setScoreRange([0, 100]); setMinTurns(1); };

  return (
    <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
      <SyntheticNote dataset={selection.dataset} />
      <header className="mb-7 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div><h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">Leads</h1><p className="mt-2 max-w-xl text-sm text-muted-foreground">Know who to follow up with first, and what temperature each conversation is at.</p></div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" onClick={() => setFilterOpen(true)} className="rounded-xl">
            <SlidersHorizontal size={16} />Filters{activeFilterCount > 0 && <Badge className="ml-1 h-5 min-w-5 justify-center px-1.5">{activeFilterCount}</Badge>}
          </Button>
          <div className="inline-flex rounded-xl bg-muted p-1" role="group" aria-label="Lead layout">
            <Toggle pressed={view === 'board'} onPressedChange={() => setView('board')} aria-label="Board view" className="h-9 gap-2 rounded-lg px-3 text-xs data-[state=on]:bg-card data-[state=on]:shadow-sm"><Columns3 size={15} /><span className="hidden sm:inline">Board</span></Toggle>
            <Toggle pressed={view === 'table'} onPressedChange={() => setView('table')} aria-label="Table view" className="h-9 gap-2 rounded-lg px-3 text-xs data-[state=on]:bg-card data-[state=on]:shadow-sm"><Table2 size={15} /><span className="hidden sm:inline">Table</span></Toggle>
          </div>
        </div>
      </header>

      <section className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card px-4 py-3 shadow-sm">
        <div className="flex items-center gap-2 text-sm font-semibold"><ListFilter size={16} className="text-primary" />{filtered.length.toLocaleString()} <span className="font-normal text-muted-foreground">of {leads.length.toLocaleString()} conversations</span></div>
        <div className="flex flex-wrap items-center gap-2">
          {statuses.map((status) => <button type="button" key={status} onClick={() => setStatuses((current) => current.filter((item) => item !== status))} className="focus-visible:ring-2 focus-visible:ring-ring"><StatusTag status={status} className="cursor-pointer" /><span className="sr-only"> remove filter</span></button>)}
          {(scoreRange[0] > 0 || scoreRange[1] < 100) && <Badge variant="secondary" className="tabular-nums">Score {scoreRange[0]}–{scoreRange[1]}%</Badge>}
          {minTurns > 1 && <Badge variant="secondary">{minTurns}+ turns</Badge>}
          {query.trim() && <Badge variant="secondary" className="max-w-48 truncate">Search: {query.trim()}</Badge>}
          {activeFilterCount > 0 && <Button variant="ghost" size="sm" onClick={clearFilters} className="h-8 px-2 text-xs text-muted-foreground">Clear all <X size={13} /></Button>}
        </div>
      </section>

      {searchError && <div role="alert" className="mb-4 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100">{searchError} Title search still works.</div>}

      {selection.evalMode && topK.length > 0 && (
        <div className="mb-5 grid gap-3 rounded-2xl border border-primary/20 bg-primary/[.04] p-4 sm:grid-cols-3">
          <Metric label="Top leads actually converted" value={`${topK.filter((lead) => lead.outcome).length} of ${topK.length}`} />
          <Metric label={`Conversion in top ${topK.length}`} value={pct(topK.filter((lead) => lead.outcome).length / topK.length)} />
          <Metric label="Base rate in this selection" value={pct(baseRate)} />
          <p className="text-xs text-muted-foreground sm:col-span-3">Evaluation mode: of the top {topK.length} leads shown, {topK.filter((lead) => lead.outcome).length} actually converted.</p>
        </div>
      )}

      {filtered.length === 0 ? (
        <Card><CardContent className="flex flex-col items-center gap-3 py-16 text-center"><EmptyIllustration /><h2 className="text-lg font-bold">No conversations match</h2><p className="text-sm text-muted-foreground">Clear the search or status filters, or widen the score range.</p><Button variant="outline" onClick={clearFilters}>Clear filters</Button></CardContent></Card>
      ) : view === 'board' ? (
        <div className="grid min-w-0 grid-cols-1 items-start gap-4 sm:grid-cols-2 xl:grid-cols-[repeat(4,minmax(0,1fr))]">
          {STATUSES.map((status) => {
            const rows = filtered.filter((lead) => lead.status === status);
            return (
              <section key={status} className="max-h-[min(72vh,760px)] min-w-0 overflow-y-auto overscroll-contain rounded-2xl border border-border bg-muted/60 p-3.5">
                <div className="sticky top-0 z-10 -mx-3.5 -mt-3.5 mb-3 flex items-center justify-between border-b border-border/70 bg-muted/95 px-4 py-3 backdrop-blur">
                  <StatusTag status={status} />
                  <Badge variant="outline" className="bg-card tabular-nums">{rows.length}</Badge>
                </div>
                <div className="grid gap-2.5">
                  {rows.map((lead) => (
                    <LeadCard key={lead.conversation_id} lead={lead} onClick={() => openConversation(lead)} />
                  ))}
                  {rows.length === 0 && <p className="rounded-xl border border-dashed border-border bg-card/60 px-3 py-6 text-center text-xs text-muted-foreground">No leads in this temperature.</p>}
                </div>
              </section>
            );
          })}
        </div>
      ) : (
        <Card className="overflow-hidden">
          <div className="max-h-[calc(100vh-310px)] min-h-96 overflow-auto">
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
                <TableRow className="hover:bg-transparent">
                  <SortHead label="Conversation" onClick={() => sortBy('title')} />
                  <SortHead label="Chance to buy" onClick={() => sortBy('final_score')} />
                  <TableHead>Trend</TableHead>
                  <SortHead label="Temperature" onClick={() => sortBy('status')} />
                  <SortHead label="Reason" onClick={() => sortBy('status_reason')} />
                  <SortHead label="Turning points" onClick={() => sortBy('n_turning_points')} />
                  <SortHead label="Turns" onClick={() => sortBy('n_turns')} />
                  {selection.evalMode && <TableHead>Outcome</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((lead) => (
                  <TableRow key={lead.conversation_id} tabIndex={0} role="button" onClick={() => openConversation(lead)}
                    onKeyDown={(event) => event.key === 'Enter' && openConversation(lead)}
                    className="cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring">
                    <TableCell className="max-w-64 font-semibold">{lead.title}</TableCell>
                    <TableCell><ScoreCell lead={lead} /></TableCell>
                    <TableCell><Sparkline values={lead.curve} width={86} height={26} /></TableCell>
                    <TableCell><StatusTag status={lead.status} /></TableCell>
                    <TableCell className="max-w-52 text-muted-foreground">{cap(lead.status_reason)}</TableCell>
                    <TableCell className="tabular-nums">{lead.n_turning_points}</TableCell>
                    <TableCell className="tabular-nums">{lead.n_turns}</TableCell>
                    {selection.evalMode && <TableCell className={lead.outcome ? 'font-semibold text-emerald-700 dark:text-emerald-300' : 'text-muted-foreground'}>{lead.outcome ? 'Converted' : 'Did not'}</TableCell>}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>
      )}

      <Sheet open={filterOpen} onOpenChange={setFilterOpen}>
        <SheetContent side="right" className="w-[min(100vw,440px)] overflow-y-auto">
          <SheetHeader>
            <SheetTitle className="flex items-center gap-2 text-xl"><Filter size={18} className="text-primary" />Filter leads</SheetTitle>
            <SheetDescription>Refine the list by temperature, score, conversation length, or words in the transcript.</SheetDescription>
          </SheetHeader>
          <div className="mt-4 space-y-6">
            <label className="block space-y-2">
              <span className="text-sm font-semibold">Search titles and messages</span>
              <span className="relative block"><Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" /><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try “discount”" aria-label="Search titles and messages" className="h-11 w-full rounded-xl border border-input bg-background pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-ring" /></span>
            </label>
            <fieldset className="space-y-3">
              <legend className="text-sm font-semibold">Temperature</legend>
              <div className="grid grid-cols-2 gap-2">{STATUSES.map((status) => <Toggle key={status} pressed={statuses.includes(status)} onPressedChange={() => setStatuses((current) => current.includes(status) ? current.filter((item) => item !== status) : [...current, status])} className="h-10 justify-start rounded-xl border border-border bg-card px-3 data-[state=on]:border-primary/50 data-[state=on]:bg-primary/10"><StatusTag status={status} /></Toggle>)}</div>
            </fieldset>
            <div className="space-y-3">
              <div className="flex justify-between text-sm font-semibold"><span>Score range</span><span className="tabular-nums text-muted-foreground">{scoreRange[0]}–{scoreRange[1]}%</span></div>
              <Slider min={0} max={100} step={5} minStepsBetweenThumbs={1} value={scoreRange} onValueChange={setScoreRange} aria-label="Score range" />
            </div>
            <label className="block space-y-2 text-sm font-semibold">Minimum turns
              <input type="number" min="1" value={minTurns} onChange={(event) => setMinTurns(Math.max(1, Number(event.target.value) || 1))} aria-label="Minimum turns" className="h-10 w-full rounded-xl border border-input bg-background px-3 text-sm font-normal outline-none focus:ring-2 focus:ring-ring" />
            </label>
            <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-border p-3 text-sm">
              <input type="checkbox" checked={selection.evalMode} onChange={(event) => selection.set({ evalMode: event.target.checked })} className="h-4 w-4 accent-primary" />
              <span><b className="block">Show true outcomes</b><small className="text-muted-foreground">Evaluation mode</small></span>
            </label>
            <div className="flex gap-2 pt-2">
              <Button variant="outline" onClick={clearFilters} className="flex-1">Clear filters</Button>
              <Button onClick={() => setFilterOpen(false)} className="flex-1">Show {filtered.length} leads</Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </main>
  );
}

function Metric({ label, value }) {
  return <div><span className="block text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</span><b className="mt-1 block tabular-nums text-xl">{value}</b></div>;
}

function LeadCard({ lead, onClick }) {
  return (
    <motion.button type="button" onClick={onClick} whileHover={{ y: -2 }} transition={{ duration: 0.15 }}
      className="group w-full min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card p-3.5 text-left shadow-sm hover:border-primary/30 hover:shadow-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-bold leading-5" title={lead.title}>{lead.title}</span>
        <ScoreCell lead={lead} compact />
      </div>
      <div className="mt-3 min-w-0">
        <Sparkline values={lead.curve} width="100%" height={30} className="!block w-full min-w-0" />
      </div>
      <Tooltip>
        <TooltipTrigger asChild>
          <span tabIndex={0} className="mt-2 block min-w-0 line-clamp-2 break-words text-left text-xs leading-5 text-muted-foreground">{cap(lead.status_reason)}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-72 leading-5">{lead.status_reason}</TooltipContent>
      </Tooltip>
      <div className="mt-2 flex min-w-0 items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <span className="shrink-0">{lead.n_turns} {lead.n_turns === 1 ? 'turn' : 'turns'}</span>
        <span className="shrink-0">{lead.n_turning_points} {lead.n_turning_points === 1 ? 'moment' : 'moments'}</span>
      </div>
    </motion.button>
  );
}

function ScoreCell({ lead, compact = false }) {
  const color = { Hot: 'from-amber-400 to-orange-500', Rising: 'from-teal-400 to-emerald-600', Cooling: 'from-sky-300 to-blue-600', Stalled: 'from-slate-400 to-slate-600' }[lead.status];
  const surface = {
    Hot: 'border-orange-200 bg-orange-100 text-orange-950 dark:border-orange-900 dark:bg-orange-950/60 dark:text-orange-100',
    Rising: 'border-teal-200 bg-teal-100 text-teal-950 dark:border-teal-900 dark:bg-teal-950/60 dark:text-teal-100',
    Cooling: 'border-sky-200 bg-sky-100 text-sky-950 dark:border-sky-900 dark:bg-sky-950/60 dark:text-sky-100',
    Stalled: 'border-slate-300 bg-slate-200 text-slate-900 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100',
  }[lead.status];
  return <div className={compact ? 'w-16 shrink-0' : 'w-28'}>
    <Badge variant="outline" className={`${surface} tabular-nums shadow-sm`}>{pct(lead.final_score)}</Badge>
    <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted"><div className={`h-full rounded-full bg-gradient-to-r ${color}`} style={{ width: pct(lead.final_score) }} /></div>
  </div>;
}

function SortHead({ label, onClick }) {
  return <TableHead><button type="button" onClick={onClick} className="inline-flex items-center gap-1.5 hover:text-foreground">{label}<ArrowDownUp size={12} /></button></TableHead>;
}

function LeadsSkeleton() {
  return <div className="mx-auto max-w-7xl space-y-5 px-5 py-8"><div className="h-10 w-52 animate-pulse rounded-lg bg-muted" /><div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">{[0, 1, 2, 3].map((i) => <div key={i} className="h-72 animate-pulse rounded-2xl bg-muted" />)}</div></div>;
}
