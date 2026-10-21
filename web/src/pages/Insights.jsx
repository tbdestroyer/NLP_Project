import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import { ArrowDownRight, ArrowUpRight, Database, Snowflake, Sparkles } from 'lucide-react';
import { getInsightExamples, getInsights } from '../api/client';
import { EmptyIllustration, ErrorNote, pct, SyntheticNote, useAsync, useSelection } from '../components/ui';
import { AGENT_MOVES, DATASETS, pretty } from '../data/taxonomy';
import {
  Badge, Button, Card, CardContent, CardHeader, CardTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
  Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, Skeleton, Table, TableBody, TableCell, TableHead, TableHeader, TableRow, Tabs, TabsContent, TabsList, TabsTrigger,
} from '../components/ui/primitives';

export default function Insights() {
  const selection = useSelection();
  const [position, setPosition] = useState('all');
  const [selectedMove, setSelectedMove] = useState('');
  const [dataOpen, setDataOpen] = useState(false);
  const [dataTab, setDataTab] = useState('examples');
  const { data: rows, error } = useAsync(() => getInsights(selection), [selection.dataset, selection.model]);
  const positions = ['all', ...(selection.server.config?.insights?.position_buckets ?? ['early', 'middle', 'late'])];
  const view = useMemo(() => (rows ?? []).filter((row) => row.position === position)
    .sort((a, b) => b.mean_delta - a.mean_delta), [rows, position]);
  const moves = view.map((row) => row.move);
  const move = moves.includes(selectedMove) ? selectedMove : moves[0];
  const exampleState = useAsync(
    () => move ? getInsightExamples({ dataset: selection.dataset, model: selection.model, move, position }) : [],
    [selection.dataset, selection.model, move, position],
  );
  const helpful = view.filter((row) => row.mean_delta > 0);
  const harmful = view.filter((row) => row.mean_delta < 0).sort((a, b) => a.mean_delta - b.mean_delta);
  const max = Math.max(0.02, ...view.map((row) => Math.max(Math.abs(row.ci_low), Math.abs(row.ci_high))));
  const minCount = selection.server.config?.insights?.min_count ?? 30;
  const agent = DATASETS[selection.dataset].agent.toLowerCase();
  const best = helpful[0];
  const worst = harmful[0];

  useEffect(() => {
    if (move && !moves.includes(selectedMove)) setSelectedMove(move);
  }, [move, selectedMove, moves.join(',')]);

  if (error) return <ErrorNote error={error} />;
  if (!rows) return <InsightsSkeleton />;

  return (
    <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
      <SyntheticNote dataset={selection.dataset} />
      <header className="mb-7 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
        <div>
          <p className="mb-2 text-xs font-bold uppercase tracking-[.16em] text-primary">Conversation patterns</p>
          <h1 className="text-3xl font-extrabold tracking-tight md:text-4xl">What moves the needle?</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">See which {agent} moves tend to come before a rise or a drop in customer inclination. These are associations, not proven causes.</p>
        </div>
        <Button variant="outline" onClick={() => setDataOpen(true)} className="w-fit rounded-xl"><Database size={16} />See data</Button>
      </header>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card p-3.5 shadow-sm">
        <div className="text-sm"><span className="font-semibold">Timing</span><span className="ml-2 text-muted-foreground">See how a move works at different points in the conversation.</span></div>
        <Tabs value={position} onValueChange={setPosition}>
          <TabsList aria-label="Position in conversation">
            {positions.map((value) => <TabsTrigger key={value} value={value}>{positionLabel(value)}</TabsTrigger>)}
          </TabsList>
        </Tabs>
      </div>

      <div className="mb-6 grid gap-3 md:grid-cols-2">
        <SummaryCard direction="up" lead={best} label={best ? `${pretty(best.move)} helps most` : 'No clear helpful move'} />
        <SummaryCard direction="down" lead={worst} label={worst ? `${pretty(worst.move)} is linked to the biggest drop` : 'No clear harmful move'} />
      </div>
      <p className="mb-5 text-xs text-muted-foreground">Test split · {DATASETS[selection.dataset].displayLabel ?? DATASETS[selection.dataset].label} · moves with fewer than {minCount} occurrences are not shown. Choose a timing tab to compare early, middle, late, or the whole conversation.</p>

      {view.length === 0 ? (
        <Card><CardContent className="flex flex-col items-center gap-3 py-16 text-center"><EmptyIllustration /><h2 className="text-lg font-bold">Not enough data here</h2><p className="text-sm text-muted-foreground">No move occurs often enough at this position. Try the whole conversation.</p></CardContent></Card>
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          <MoveColumn title="What helps" description="Moves associated with the customer's score rising." icon={ArrowUpRight} tone="text-teal-700 dark:text-teal-200" rows={helpful} max={max} positive />
          <MoveColumn title="What hurts" description="Moves associated with the customer's score dropping." icon={ArrowDownRight} tone="text-orange-700 dark:text-orange-200" rows={harmful} max={max} />
        </div>
      )}

      <p className="mt-5 text-xs leading-5 text-muted-foreground">Average change is measured on the customer's next message. The whisker shows the 95% confidence interval; hover a bar for the exact interval and how often the move was followed by a rise or drop.</p>

      <Sheet open={dataOpen} onOpenChange={setDataOpen}>
        <SheetContent side="right" className="w-[min(100vw,900px)] overflow-hidden sm:max-w-[900px]">
          <SheetHeader>
            <SheetTitle className="text-xl">Insights data</SheetTitle>
            <SheetDescription>Explore the underlying values and transcript examples used in this view.</SheetDescription>
          </SheetHeader>
          <Tabs value={dataTab} onValueChange={setDataTab} className="mt-3 flex min-h-0 flex-1 flex-col overflow-hidden">
            <TabsList className="w-fit"><TabsTrigger value="examples">Examples</TabsTrigger><TabsTrigger value="table">Table</TabsTrigger></TabsList>
            <TabsContent value="table" className="min-h-0 flex-1 overflow-auto rounded-xl border border-border">
              <Table>
                <TableHeader className="sticky top-0 z-10 bg-muted"><TableRow>
                  <TableHead>Agent move</TableHead><TableHead>Average change</TableHead><TableHead>95% CI low</TableHead><TableHead>95% CI high</TableHead>
                  <TableHead>Occurrences</TableHead><TableHead>Followed by rise</TableHead><TableHead>Followed by drop</TableHead>
                </TableRow></TableHeader>
                <TableBody>{view.map((row) => (
                  <TableRow key={row.move}>
                    <TableCell className="font-semibold">{pretty(row.move)}</TableCell>
                    <TableCell className={`tabular-nums font-bold ${row.mean_delta >= 0 ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}`}>{row.mean_delta >= 0 ? '+' : '−'}{Math.abs(row.mean_delta * 100).toFixed(2)} pts</TableCell>
                    <TableCell className="tabular-nums">{(row.ci_low * 100).toFixed(2)}</TableCell><TableCell className="tabular-nums">{(row.ci_high * 100).toFixed(2)}</TableCell>
                    <TableCell className="tabular-nums">{row.count}</TableCell><TableCell className="tabular-nums">{pct(row.share_rise)}</TableCell><TableCell className="tabular-nums">{pct(row.share_drop)}</TableCell>
                  </TableRow>
                ))}</TableBody>
              </Table>
            </TabsContent>
            <TabsContent value="examples" className="min-h-0 flex-1 overflow-y-auto">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border p-3">
                <div><p className="text-sm font-bold">Move examples</p><p className="text-xs text-muted-foreground">Agent message and the customer's next reply.</p></div>
                <Select value={move ?? ''} onValueChange={setSelectedMove}>
                  <SelectTrigger className="w-56"><SelectValue placeholder="Choose a move" /></SelectTrigger>
                  <SelectContent>{moves.map((key) => <SelectItem key={key} value={key}>{pretty(key)}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              {exampleState.error && <div role="alert" className="rounded-xl border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900">{exampleState.error.message}</div>}
              {exampleState.loading && <div className="space-y-3"><Skeleton className="h-36" /><Skeleton className="h-36" /></div>}
              {exampleState.data?.length === 0 && !exampleState.loading && <p className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">No examples are available for this move and position.</p>}
              <div className="grid gap-3">{exampleState.data?.map((example) => <ExampleCard key={`${example.conversation_id}-${example.agent_turn}`} example={example} onClose={() => setDataOpen(false)} />)}</div>
            </TabsContent>
          </Tabs>
        </SheetContent>
      </Sheet>
    </main>
  );
}

function SummaryCard({ direction, lead, label }) {
  const positive = direction === 'up';
  const Icon = positive ? Sparkles : Snowflake;
  return (
    <Card className="overflow-hidden">
      <CardContent className="flex items-center gap-4 p-4">
        <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-xl ${positive ? 'bg-teal-50 text-teal-700 dark:bg-teal-950/50 dark:text-teal-200' : 'bg-orange-50 text-orange-700 dark:bg-orange-950/50 dark:text-orange-200'}`}><Icon size={19} /></span>
        <div className="min-w-0"><p className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">{positive ? 'Biggest lift' : 'Biggest drop'}</p><p className="mt-0.5 truncate text-sm font-bold">{label}</p></div>
        {lead && <span className={`ml-auto shrink-0 tabular-nums text-lg font-extrabold ${positive ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}`}>{lead.mean_delta >= 0 ? '+' : '−'}{Math.abs(lead.mean_delta * 100).toFixed(1)} pts</span>}
      </CardContent>
    </Card>
  );
}

function MoveColumn({ title, description, icon: Icon, tone, rows, max, positive = false }) {
  const chartData = rows.map((row) => {
    const value = Math.abs(row.mean_delta);
    const low = positive ? Math.max(0, row.ci_low) : Math.max(0, Math.abs(row.ci_high));
    const high = positive ? Math.max(value, row.ci_high) : Math.max(value, Math.abs(row.ci_low));
    return {
      label: pretty(row.move),
      value: value * 100,
      error: [Math.max(0, (value - low) * 100), Math.max(0, (high - value) * 100)],
    };
  });
  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border/70 pb-4">
        <CardTitle className="flex items-center gap-2 text-base"><Icon size={19} className={tone} />{title}</CardTitle>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardHeader>
      <CardContent className="space-y-2 p-4">
        {rows.length === 0 && <p className="rounded-xl bg-muted/70 p-4 text-sm text-muted-foreground">No moves in this group for the selected timing.</p>}
        {rows.length > 0 && <div className="mb-3 h-[min(40vh,340px)] min-h-40 w-full" aria-label={`${title} agent move effects`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 18, bottom: 4, left: 4 }} barCategoryGap={10}>
              <CartesianGrid horizontal={false} stroke="hsl(var(--border))" />
              <XAxis type="number" domain={[0, Math.max(1, max * 100)]} hide />
              <YAxis type="category" dataKey="label" width={128} tickLine={false} axisLine={false} tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }} />
              <ChartTooltip cursor={{ fill: 'hsl(var(--muted))', opacity: 0.5 }} formatter={(value) => [`${positive ? '+' : '−'}${Number(value).toFixed(1)} pts`, 'Average change']} />
              <Bar dataKey="value" radius={[0, 7, 7, 0]} maxBarSize={18} isAnimationActive animationDuration={220}>
                {chartData.map((item) => <Cell key={item.label} fill={positive ? '#0f9f88' : '#f97316'} />)}
                <ErrorBar dataKey="error" direction="x" width={6} stroke="hsl(var(--foreground))" strokeWidth={1.3} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>}
        {rows.map((row) => <MoveCard key={row.move} row={row} max={max} positive={positive} />)}
      </CardContent>
    </Card>
  );
}

function MoveCard({ row, max, positive }) {
  const low = positive ? Math.max(0, row.ci_low) : Math.max(0, Math.abs(row.ci_high));
  const high = positive ? Math.min(max, row.ci_high) : Math.min(max, Math.abs(row.ci_low));
  const value = Math.abs(row.mean_delta);
  const fillStart = Math.max(0, Math.min(100, (low / max) * 100));
  const fillWidth = Math.max(3, Math.min(100 - fillStart, ((high - low) / max) * 100));
  const color = positive ? '#0f9f88' : '#f97316';
  return (
    <div className="rounded-xl border border-border/80 p-3.5 transition-colors hover:bg-muted/50">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0"><p className="text-sm font-bold">{pretty(row.move)}</p><p className="mt-0.5 text-xs text-muted-foreground">{AGENT_MOVES[row.move]?.phrase}</p></div>
        <div className="text-right"><p className={`tabular-nums text-lg font-extrabold ${positive ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}`}>{positive ? '+' : '−'}{(value * 100).toFixed(1)} pts</p><p className="text-[10px] text-muted-foreground">{row.count} times</p></div>
      </div>
      <div className="mt-3 space-y-1">
        <div className="relative h-2 overflow-hidden rounded-full bg-muted" title={`95% confidence interval ${row.ci_low * 100} to ${row.ci_high * 100} points`}>
          <span className="absolute inset-y-0 rounded-full" style={{ left: `${fillStart}%`, width: `${fillWidth}%`, backgroundColor: color }} />
          <span className="absolute top-[-3px] h-4 w-px bg-foreground/70" style={{ left: `${Math.min(99, Math.max(1, (value / max) * 100))}%` }} />
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground"><span>95% confidence range</span><span>{pct(row.share_rise)} rise · {pct(row.share_drop)} drop</span></div>
      </div>
    </div>
  );
}

function ExampleCard({ example, onClose }) {
  const location = useLocation();
  return (
    <article className="rounded-2xl border border-border bg-card p-4">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <b className="text-sm">Score change {example.delta >= 0 ? '+' : '−'}{Math.abs(example.delta * 100).toFixed(0)} pts on customer's reply</b>
        <Link to={`/conversations/${encodeURIComponent(example.conversation_id)}`} state={{ backgroundLocation: location }} onClick={onClose} className="text-xs font-bold text-primary hover:underline">Open conversation →</Link>
      </header>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-xl bg-muted/70 p-3"><p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Sales side · turn {example.agent_turn}</p><p className="text-sm leading-6">{example.agent_text}</p></div>
        <div className={`rounded-xl p-3 ${example.delta >= 0 ? 'bg-teal-50 dark:bg-teal-950/40' : 'bg-orange-50 dark:bg-orange-950/40'}`}><p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Customer · turn {example.customer_turn}</p><p className="text-sm leading-6">{example.customer_text}</p></div>
      </div>
    </article>
  );
}

function InsightsSkeleton() {
  return <div className="mx-auto max-w-7xl space-y-5 px-5 py-8"><Skeleton className="h-12 w-72" /><div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-96" /><Skeleton className="h-96" /></div></div>;
}

function positionLabel(position) {
  return ({ all: 'Whole conversation', early: 'Early', middle: 'Middle', late: 'Late' })[position] ?? position;
}
