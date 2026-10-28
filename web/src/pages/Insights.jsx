import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ArrowDownRight, ArrowUpRight, Database } from 'lucide-react';
import { getInsightExamples, getInsights } from '../api/client';
import { EmptyIllustration, ErrorNote, pct, SyntheticNote, useAsync, useSelection } from '../components/ui';
import { DATASETS, pretty } from '../data/taxonomy';
import { ExampleCard, MoveColumn, positionLabel, splitEffects, SummaryCard } from '../components/InsightViews';
import {
  Badge, Button, Card, CardContent, CardHeader, CardTitle, Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
  Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, Skeleton, Table, TableBody, TableCell, TableHead, TableHeader, TableRow, Tabs, TabsContent, TabsList, TabsTrigger,
} from '../components/ui/primitives';

export default function Insights() {
  const selection = useSelection();
  const location = useLocation();
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
  const { helpful, harmful, max } = splitEffects(view);
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
              <div className="grid gap-3">{exampleState.data?.map((example) => <ExampleCard key={`${example.conversation_id}-${example.agent_turn}`} example={example}
                action={<Link to={`/conversations/${encodeURIComponent(example.conversation_id)}`} state={{ backgroundLocation: location }} onClick={() => setDataOpen(false)} className="text-xs font-bold text-primary hover:underline">Open conversation →</Link>} />)}</div>
            </TabsContent>
          </Tabs>
        </SheetContent>
      </Sheet>
    </main>
  );
}

function InsightsSkeleton() {
  return <div className="mx-auto max-w-7xl space-y-5 px-5 py-8"><Skeleton className="h-12 w-72" /><div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-96" /><Skeleton className="h-96" /></div></div>;
}
