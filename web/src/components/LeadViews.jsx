import { motion } from 'framer-motion';
import { ArrowDownUp } from 'lucide-react';
import { cap, pct, Sparkline } from './ui';
import { Badge, TableHead, Tooltip, TooltipContent, TooltipTrigger } from './ui/primitives';

/** Lead cards, score cells and sortable headers shared by the Leads page and batch analysis. */
export function LeadCard({ lead, onClick, rank }) {
  return (
    <motion.button type="button" onClick={onClick} whileHover={{ y: -2 }} transition={{ duration: 0.15 }}
      className="group w-full min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card p-3.5 text-left shadow-sm hover:border-primary/30 hover:shadow-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-bold leading-5" title={lead.title}>{rank != null && <span className="mr-1.5 tabular-nums text-muted-foreground">#{rank}</span>}{lead.title}</span>
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

export function ScoreCell({ lead, compact = false }) {
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

export function SortHead({ label, onClick }) {
  return <TableHead><button type="button" onClick={onClick} className="inline-flex items-center gap-1.5 hover:text-foreground">{label}<ArrowDownUp size={12} /></button></TableHead>;
}
