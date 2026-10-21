import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, CircleHelp, Flame, MessageSquareText, Snowflake, Target, ThermometerSun, TrendingUp } from 'lucide-react';
import { getLeads, getMetrics } from '../api/client';
import { EmptyIllustration, ErrorNote, pct, Sparkline, StatusTag, SyntheticNote, useAsync, useSelection } from '../components/ui';
import { DATASETS, STATUSES } from '../data/taxonomy';
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Skeleton, Tooltip, TooltipContent, TooltipTrigger } from '../components/ui/primitives';

const COLUMN_META = {
  Hot: { icon: Flame, tone: 'text-orange-700 bg-orange-100 dark:text-orange-200 dark:bg-orange-950/50', description: 'Ready to make a move' },
  Rising: { icon: TrendingUp, tone: 'text-teal-700 bg-teal-100 dark:text-teal-200 dark:bg-teal-950/50', description: 'Momentum is building' },
  Stalled: { icon: ThermometerSun, tone: 'text-slate-700 bg-slate-200 dark:text-slate-200 dark:bg-slate-800', description: 'Needs a fresh approach' },
  Cooling: { icon: Snowflake, tone: 'text-sky-800 bg-sky-100 dark:text-sky-200 dark:bg-sky-950/50', description: 'May need re-engagement' },
};

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.045 } } };
const riseIn = { hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: { duration: 0.2 } } };

export default function Overview() {
  const selection = useSelection();
  const leadState = useAsync(() => getLeads(selection), [selection.dataset, selection.model, selection.split]);
  const metricState = useAsync(() => getMetrics(selection), [selection.dataset, selection.model]);
  const location = useLocation();
  const navigate = useNavigate();

  if (leadState.error) return <ErrorNote error={leadState.error} />;
  if (metricState.error) return <ErrorNote error={metricState.error} />;
  if (!leadState.data || metricState.loading) {
    return <div className="mx-auto grid max-w-7xl gap-5 px-5 py-8"><Skeleton className="h-48" /><div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{[0, 1, 2, 3].map((index) => <Skeleton key={index} className="h-32" />)}</div><Skeleton className="h-96" /></div>;
  }

  const leads = leadState.data;
  const metrics = metricState.data;
  const average = leads.length ? leads.reduce((sum, lead) => sum + lead.final_score, 0) / leads.length : null;
  const curveLength = Math.max(0, ...leads.map((lead) => lead.curve?.length ?? 0));
  const averageCurve = Array.from({ length: curveLength }, (_, turn) => {
    const values = leads.map((lead) => lead.curve?.[turn]).filter(Number.isFinite);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }).filter(Number.isFinite);
  const hot = leads.filter((lead) => lead.status === 'Hot').length;
  const datasetLabel = DATASETS[selection.dataset]?.displayLabel ?? DATASETS[selection.dataset]?.label ?? selection.dataset;
  const temperature = STATUSES.map((status) => ({
    status,
    leads: [...leads].filter((lead) => lead.status === status).sort((a, b) => b.final_score - a.final_score),
  }));

  return (
    <main className="mx-auto max-w-[1460px] px-5 py-7 md:px-8 md:py-9">
      <SyntheticNote dataset={selection.dataset} />
      <section className="relative isolate overflow-hidden rounded-[28px] bg-gradient-to-br from-navy via-blue-700 to-sky-500 px-6 py-8 text-white shadow-float md:px-10 md:py-10">
        <div aria-hidden="true" className="absolute -right-12 -top-28 -z-10 h-80 w-80 rounded-full border border-white/15 shadow-[0_0_0_36px_rgb(255_255_255/6%),0_0_0_72px_rgb(255_255_255/5%)]" />
        <p className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-[.2em] text-blue-100"><span className="h-1.5 w-1.5 rounded-full bg-sky-200" />Sales pipeline</p>
        <div className="flex flex-col justify-between gap-7 md:flex-row md:items-end">
          <div className="max-w-3xl">
            <h1 className="text-balance text-3xl font-extrabold leading-tight tracking-[-0.05em] md:text-5xl">
              {hot.toLocaleString()} of {leads.length.toLocaleString()} {leads.length === 1 ? 'lead is' : 'leads are'} ready to close
            </h1>
            <p className="mt-4 text-sm font-medium text-blue-50 md:text-base">{datasetLabel} <span className="px-1.5 text-white/60">·</span> {selection.model} model <span className="px-1.5 text-white/60">·</span> {selection.split} split</p>
          </div>
          <Button asChild variant="secondary" className="w-fit rounded-full border border-white/30 bg-white/10 text-white hover:bg-white/20">
            <a href="#/leads">Explore leads <ArrowRight size={16} /></a>
          </Button>
        </div>
      </section>

      <section aria-label="Pipeline stats" className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={MessageSquareText} label="Conversations scored" value={leads.length} spark={averageCurve} />
        <StatCard icon={ThermometerSun} label="Average chance to buy" value={average == null ? 'n/a' : average * 100} suffix={average == null ? '' : '%'} decimals={0} spark={averageCurve} />
        <StatCard icon={Flame} label="Hot leads" value={hot} spark={averageCurve} />
        <StatCard icon={Target} label="Test ROC AUC" value={metrics?.roc_auc ?? 'n/a'} decimals={3}
          spark={averageCurve}
          hint={metrics ? `How well final scores rank buyers above non-buyers. 95% interval ${metrics.roc_auc_ci[0].toFixed(3)}–${metrics.roc_auc_ci[1].toFixed(3)}.` : 'Run make eval to compute this metric.'} />
      </section>

      <div className="mb-4 mt-10 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Lead temperature</h2>
          <p className="mt-1 text-sm text-muted-foreground">See who is warming up, cooling off, and ready for a follow-up.</p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-2 text-xs font-semibold text-muted-foreground">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />{leads.length.toLocaleString()} conversations
        </div>
      </div>

      {leads.length ? (
        <motion.section variants={stagger} initial="hidden" animate="show" aria-label="Leads by temperature"
          className="grid min-w-0 grid-cols-1 items-start gap-4 sm:grid-cols-2 xl:grid-cols-[repeat(4,minmax(0,1fr))]">
          {temperature.map(({ status, leads: columnLeads }) => {
            const meta = COLUMN_META[status];
            const StatusIcon = meta.icon;
            return (
              <motion.section variants={riseIn} key={status} className="max-h-[min(72vh,760px)] min-w-0 overflow-y-auto overscroll-contain rounded-2xl border border-border bg-muted/60 p-3.5">
                <header className="sticky top-0 z-10 -mx-3.5 -mt-3.5 mb-3 flex items-start justify-between gap-2 border-b border-border/70 bg-muted/95 px-3.5 py-3 backdrop-blur">
                  <div className="flex items-center gap-2.5">
                    <span className={`grid h-9 w-9 place-items-center rounded-xl ${meta.tone}`}><StatusIcon size={17} /></span>
                    <div><h3 className="text-sm font-bold">{status}</h3><p className="text-[11px] text-muted-foreground">{meta.description}</p></div>
                  </div>
                  <Badge variant="outline" className="border-border bg-card tabular-nums">{columnLeads.length}</Badge>
                </header>
                <div className="grid gap-2.5">
                  {columnLeads.slice(0, 8).map((lead) => (
                    <motion.button variants={riseIn} key={lead.conversation_id} type="button"
                      onClick={() => navigate(`/conversations/${encodeURIComponent(lead.conversation_id)}`, { state: { backgroundLocation: location } })}
                      className="group w-full min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card p-3.5 text-left shadow-sm transition-all duration-150 hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                      <div className="flex min-w-0 items-center justify-between gap-2">
                        <p title={lead.title} className="min-w-0 flex-1 truncate text-sm font-bold leading-5">{lead.title}</p>
                        <Badge variant="outline" className={`shrink-0 border-0 px-2.5 py-1 tabular-nums ${scoreClasses(status)}`}>{pct(lead.final_score)}</Badge>
                      </div>
                      <div className="mt-3 min-w-0">
                        <Sparkline values={lead.curve} width="100%" height={30} className="!block w-full min-w-0" />
                      </div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span tabIndex={0} className="mt-2 block min-w-0 line-clamp-2 break-words text-left text-xs leading-5 text-muted-foreground">{lead.status_reason}</span>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-72 leading-5">{lead.status_reason}</TooltipContent>
                      </Tooltip>
                      <div className="mt-2 flex min-w-0 items-center justify-between gap-2 text-[11px] text-muted-foreground">
                        <span className="shrink-0">{lead.n_turns} {lead.n_turns === 1 ? 'turn' : 'turns'}</span>
                        <span className="shrink-0">{lead.n_turning_points} {lead.n_turning_points === 1 ? 'moment' : 'moments'}</span>
                      </div>
                    </motion.button>
                  ))}
                  {columnLeads.length === 0 && <p className="rounded-xl border border-dashed border-border bg-card/60 px-3 py-6 text-center text-xs text-muted-foreground">No leads in this temperature.</p>}
                  {columnLeads.length > 8 && <Button asChild variant="ghost" size="sm" className="w-full text-muted-foreground"><a href="#/leads">View all {columnLeads.length} <ArrowRight size={14} /></a></Button>}
                </div>
              </motion.section>
            );
          })}
        </motion.section>
      ) : (
        <Card><CardContent className="flex flex-col items-center gap-3 py-16 text-center"><EmptyIllustration /><p className="font-semibold">No conversations in this selection.</p></CardContent></Card>
      )}
      <p className="mt-6 text-xs leading-5 text-muted-foreground">The datasets are proxies for sales: price negotiation (CraigslistBargains), charity donation (Persuasion for Good), and synthetic SaaS sales chats. Everything shown is precomputed.</p>
    </main>
  );
}

function scoreClasses(status) {
  return {
    Hot: 'bg-orange-100 text-orange-900 dark:bg-orange-950/60 dark:text-orange-100',
    Rising: 'bg-teal-100 text-teal-900 dark:bg-teal-950/60 dark:text-teal-100',
    Stalled: 'bg-slate-200 text-slate-800 dark:bg-slate-800 dark:text-slate-100',
    Cooling: 'bg-sky-100 text-sky-900 dark:bg-sky-950/60 dark:text-sky-100',
  }[status];
}

function StatCard({ icon: Icon, label, value, suffix = '', decimals = 0, hint, spark = [] }) {
  const [display, setDisplay] = useState(typeof value === 'number' ? 0 : value);
  useEffect(() => {
    if (typeof value !== 'number') { setDisplay(value); return undefined; }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setDisplay(value); return undefined; }
    let frame;
    const start = performance.now();
    const animate = (now) => {
      const progress = Math.min(1, (now - start) / 520);
      setDisplay(value * (1 - (1 - progress) ** 3));
      if (progress < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  const formatted = typeof value !== 'number' ? value : decimals ? display.toFixed(decimals) : Math.round(display).toLocaleString();
  return (
    <Card title={hint} className="min-w-0 transition-shadow hover:shadow-float">
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-1">
        <CardTitle className="text-xs font-semibold text-muted-foreground">{label}</CardTitle>
        <span className="grid h-9 w-9 place-items-center rounded-xl bg-primary/10 text-primary"><Icon size={17} /></span>
      </CardHeader>
      <CardContent>
        <div className="flex items-end justify-between gap-2">
          <p className="tabular-nums text-3xl font-extrabold tracking-tight">{formatted}{suffix}</p>
          {spark.length > 1 && <Sparkline values={spark} width={72} height={30} />}
        </div>
          {hint && <div className="mt-1 flex items-center gap-1 text-[11px] text-muted-foreground"><span className="line-clamp-1">{hint}</span><Tooltip><TooltipTrigger asChild><button type="button" aria-label={`More about ${label}`} className="shrink-0 rounded-full p-1 hover:bg-muted"><CircleHelp size={13} /></button></TooltipTrigger><TooltipContent className="max-w-72 leading-5">{hint}</TooltipContent></Tooltip></div>}
      </CardContent>
    </Card>
  );
}
