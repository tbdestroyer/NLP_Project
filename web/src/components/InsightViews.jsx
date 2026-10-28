import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import { Snowflake, Sparkles } from 'lucide-react';
import { pct } from './ui';
import { AGENT_MOVES, pretty } from '../data/taxonomy';
import { Card, CardContent, CardHeader, CardTitle } from './ui/primitives';

/** Script-insight cards and charts shared by the Insights page and batch analysis. */
export function SummaryCard({ direction, lead, label }) {
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

export function MoveColumn({ title, description, icon: Icon, tone, rows, max, positive = false }) {
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

/** One agent message and the customer's reply; `action` is the link or button that opens it. */
export function ExampleCard({ example, action }) {
  return (
    <article className="rounded-2xl border border-border bg-card p-4">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <b className="text-sm">Score change {example.delta >= 0 ? '+' : '−'}{Math.abs(example.delta * 100).toFixed(0)} pts on customer's reply</b>
        {action}
      </header>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-xl bg-muted/70 p-3"><p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Sales side · turn {example.agent_turn}</p><p className="text-sm leading-6">{example.agent_text}</p></div>
        <div className={`rounded-xl p-3 ${example.delta >= 0 ? 'bg-teal-50 dark:bg-teal-950/40' : 'bg-orange-50 dark:bg-orange-950/40'}`}><p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Customer · turn {example.customer_turn}</p><p className="text-sm leading-6">{example.customer_text}</p></div>
      </div>
    </article>
  );
}

export function positionLabel(position) {
  return ({ all: 'Whole conversation', early: 'Early', middle: 'Middle', late: 'Late' })[position] ?? position;
}

/** Splits move effects into helpful (rising, biggest first) and harmful (biggest drop first). */
export function splitEffects(view) {
  const helpful = view.filter((row) => row.mean_delta > 0).sort((a, b) => b.mean_delta - a.mean_delta);
  const harmful = view.filter((row) => row.mean_delta < 0).sort((a, b) => a.mean_delta - b.mean_delta);
  const max = Math.max(0.02, ...view.map((row) => Math.max(Math.abs(row.ci_low), Math.abs(row.ci_high))));
  return { helpful, harmful, max };
}
