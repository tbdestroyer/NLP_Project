import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import { ArrowDownRight, ArrowUpRight, ChevronDown, CircleHelp, Pause, Play, Sparkles, Target } from 'lucide-react';
import { cap, pct, pts, StatusTag } from './ui';
import { CUSTOMER_CUES, DATASETS, MODELS, pretty } from '../data/taxonomy';
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Slider } from './ui/primitives';

function CueChips({ cues }) {
  if (!cues?.length) return null;
  return <div className="mt-2 flex flex-wrap gap-1.5">{cues.map((cue) => {
    const direction = CUSTOMER_CUES[cue]?.direction;
    return <Badge key={cue} variant="outline" className={`border-0 px-2 py-0.5 text-[10px] ${direction > 0 ? 'bg-teal-50 text-teal-800 dark:bg-teal-950/50 dark:text-teal-100' : direction < 0 ? 'bg-orange-50 text-orange-800 dark:bg-orange-950/50 dark:text-orange-100' : 'bg-muted text-muted-foreground'}`}>{pretty(cue)}</Badge>;
  })}</div>;
}

function topFeatures(value) {
  if (!value) return [];
  return typeof value === 'string' ? JSON.parse(value) : value;
}

function ChartHint({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-xl border border-border bg-card px-3 py-2 text-xs shadow-float">
      <p className="font-bold">Turn {point.turn} · {point.speaker || cap(point.role)}</p>
      <p className="mt-1 tabular-nums text-primary">Chance to buy: {pct(point.score)}</p>
      {point.delta != null && <p className={point.delta >= 0 ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}>{pts(point.delta)} from previous turn</p>}
    </div>
  );
}

/** Transcript, score curve, replay and turning-point evidence shared by real and preview data. */
export default function ConversationDetail({ convo, showOutcome = false, model = 'v1', split = 'test' }) {
  const [active, setActive] = useState(null);
  const [replay, setReplay] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [pointTurn, setPointTurn] = useState(null);
  const [showEvidence, setShowEvidence] = useState(true);
  const refs = useRef({});
  const n = convo.turns.length;
  const names = DATASETS[convo.dataset] ?? { customer: 'Customer', agent: 'Agent' };
  const shownUpTo = replay ?? n - 1;
  const visiblePoints = convo.turning_points.filter((point) => point.turn <= shownUpTo);
  const pointByTurn = Object.fromEntries(visiblePoints.map((point) => [point.turn, point]));
  const selectedPoint = visiblePoints.find((point) => point.turn === pointTurn) ?? visiblePoints[0];
  const score = convo.scores[shownUpTo];
  const previousScore = shownUpTo > 0 ? convo.scores[shownUpTo - 1] : null;
  const selectedModel = MODELS[model]?.label ?? model;
  const overlays = Object.entries(convo.scores_by_model ?? {}).filter(([name]) => name !== model)
    .map(([name, scores]) => ({ name: MODELS[name]?.label ?? name, scores }));
  const chartData = useMemo(() => convo.turns.slice(0, shownUpTo + 1).map((turn, index) => ({
    turn: turn.turn,
    score: convo.scores[index],
    role: turn.role,
    speaker: turn.speaker,
    delta: index ? convo.scores[index] - convo.scores[index - 1] : null,
    point: pointByTurn[turn.turn],
  })), [convo, shownUpTo, visiblePoints.map((point) => point.turn).join(',')]);

  useEffect(() => {
    if (!playing) return undefined;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const interval = setInterval(() => {
      setReplay((current) => {
        const next = (current ?? -1) + 1;
        if (next >= n - 1) { setPlaying(false); return n - 1; }
        return next;
      });
    }, reduced ? 700 : 480);
    return () => clearInterval(interval);
  }, [playing, n]);

  useEffect(() => { if (replay != null) setActive(replay); }, [replay]);
  useEffect(() => {
    if (!visiblePoints.some((point) => point.turn === pointTurn)) setPointTurn(visiblePoints[0]?.turn ?? null);
  }, [pointTurn, visiblePoints.map((point) => point.turn).join(',')]);

  const focusTurn = (index) => {
    setActive(index);
    if (pointByTurn[index]) setPointTurn(index);
    refs.current[index]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  const handleReplay = () => {
    if (playing) { setPlaying(false); return; }
    if (replay == null || replay >= n - 1) setReplay(0);
    setPlaying(true);
  };
  const features = selectedPoint ? topFeatures(selectedPoint.top_features) : [];

  return (
    <div className="mx-auto max-w-[1320px] space-y-5 px-5 py-6 md:px-8 md:py-8">
      <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{names.displayLabel ?? names.label ?? convo.dataset}</Badge>
            <Badge variant="outline">{n} turns</Badge><Badge variant="outline">{selectedModel}</Badge><Badge variant="outline">{split} split</Badge>
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight md:text-3xl">{convo.title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{cap(convo.status_reason)}.</p>
          {model === 'v2' && <p className="mt-2 max-w-3xl rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-950 dark:bg-amber-950/40 dark:text-amber-100">v2 scores come from Gemini reading the conversation so far. They are not calibrated and exist for a small test sample only.</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2"><StatusTag status={convo.status} /><Badge variant="outline" className="bg-card tabular-nums">{pct(convo.final_score)} chance to buy</Badge></div>
      </header>

      <section className="grid gap-3 sm:grid-cols-3">
        <MetricCard label="Final score" value={pct(convo.final_score)} icon={Target} />
        <MetricCard label="Lead temperature" value={convo.status} icon={convo.status === 'Hot' ? Sparkles : ArrowUpRight} secondary={cap(convo.status_reason)} />
        <MetricCard label={showOutcome ? 'True outcome' : 'Turning points'} value={showOutcome ? convo.outcome ? 'Converted' : 'Did not convert' : convo.n_turning_points} icon={showOutcome ? Target : ArrowDownRight} />
      </section>

      <Card className="overflow-hidden">
        <CardHeader className="flex-row flex-wrap items-start justify-between gap-3 border-b border-border/70 pb-4">
          <div><CardTitle>Chance to buy after each message</CardTitle><p className="mt-1 text-xs text-muted-foreground">Filled dots are {names.customer.toLowerCase()} messages. Markers show turning points.</p></div>
          <div className="flex items-center gap-2">
            <span className={`text-sm font-extrabold tabular-nums ${score >= 0.7 ? 'text-orange-700 dark:text-orange-200' : 'text-primary'}`}>{pct(score)}</span>
            {previousScore != null && <Badge variant="outline" className={score >= previousScore ? 'border-teal-200 text-teal-800 dark:border-teal-900 dark:text-teal-200' : 'border-orange-200 text-orange-800 dark:border-orange-900 dark:text-orange-200'}>{pts(score - previousScore)}</Badge>}
          </div>
        </CardHeader>
        <CardContent className="px-2 pb-3 pt-5 sm:px-5">
          <div className="h-[260px] w-full sm:h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 18, right: 18, left: -18, bottom: 4 }}>
                <defs><linearGradient id={`inclination-fill-${convo.conversation_id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2563eb" stopOpacity={0.28} /><stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} /></linearGradient></defs>
                <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="4 5" />
                <XAxis dataKey="turn" tickLine={false} axisLine={false} fontSize={11} tick={{ fill: 'hsl(var(--muted-foreground))' }} />
                <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} fontSize={11} tick={{ fill: 'hsl(var(--muted-foreground))' }} width={42} />
                <ReferenceLine y={0.7} stroke="#0f9f88" strokeDasharray="5 5" strokeOpacity={0.65} />
                <ChartTooltip content={<ChartHint />} />
                <Area type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2.5} fill={`url(#inclination-fill-${convo.conversation_id})`} dot={(props) => {
                  const dataPoint = props.payload;
                  if (!dataPoint?.point) return <circle key={`dot-${dataPoint?.turn}`} cx={props.cx} cy={props.cy} r={dataPoint?.role === 'customer' ? 3 : 1.8} fill={dataPoint?.role === 'customer' ? '#2563eb' : 'hsl(var(--card))'} stroke="#2563eb" strokeWidth={1.5} />;
                  const up = dataPoint.point.direction === 'rise';
                  const color = up ? '#f59e0b' : '#0ea5e9';
                  return <g key={`point-${dataPoint.turn}`} role="button" tabIndex={0} aria-label={`Turn ${dataPoint.turn}, ${up ? 'rise' : 'drop'} ${pts(dataPoint.point.delta)}`} onClick={() => focusTurn(dataPoint.turn)} onKeyDown={(event) => event.key === 'Enter' && focusTurn(dataPoint.turn)} className="cursor-pointer">
                    <circle cx={props.cx} cy={props.cy} r="10" fill={color} opacity="0.15" /><circle cx={props.cx} cy={props.cy} r="5.5" fill={color} stroke="white" strokeWidth="2" />
                    <text x={props.cx + 8} y={props.cy - 10} fill={color} fontSize="10" fontWeight="700">{dataPoint.point.delta >= 0 ? '+' : '−'}{Math.abs(dataPoint.point.delta * 100).toFixed(0)} pts</text>
                  </g>;
                }} isAnimationActive={replay != null} animationDuration={280} activeDot={{ r: 5, strokeWidth: 2 }} />
                {overlays.map((overlay, index) => <Area key={overlay.name} type="monotone" data={overlay.scores.slice(0, shownUpTo + 1).map((value, turn) => ({ turn, score: value }))} dataKey="score" name={overlay.name} stroke={index % 2 ? '#c2410c' : '#7c3aed'} strokeDasharray="5 4" fill="none" strokeWidth={1.5} dot={false} isAnimationActive={false} />)}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          {overlays.length > 0 && <div className="mt-1 flex flex-wrap justify-end gap-3 text-[11px] text-muted-foreground">{[selectedModel, ...overlays.map((overlay) => overlay.name)].map((name, index) => <span key={name} className="inline-flex items-center gap-1.5"><i className="h-0.5 w-4 rounded" style={{ backgroundColor: index === 0 ? '#2563eb' : index % 2 ? '#7c3aed' : '#c2410c' }} />{name}</span>)}</div>}
          {n > 1 && (
            <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-border/70 pt-4">
              <Button variant="outline" size="sm" onClick={handleReplay} className="rounded-xl"><span className="grid h-5 w-5 place-items-center rounded-full bg-primary/10 text-primary">{playing ? <Pause size={11} /> : <Play size={11} />}</span>{playing ? 'Pause replay' : 'Replay'}</Button>
              <label className="min-w-52 flex-1 text-xs font-semibold text-muted-foreground">Replay up to turn {shownUpTo}<Slider min={0} max={n - 1} step={1} value={[shownUpTo]} onValueChange={(value) => { setPlaying(false); setReplay(value[0]); }} aria-label="Replay: show conversation up to turn" className="mt-1" /></label>
              <Button variant="ghost" size="sm" onClick={() => { setReplay(null); setPlaying(false); }} className="text-xs text-muted-foreground">Show full conversation</Button>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(360px,.8fr)]">
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-border/70 pb-4"><CardTitle>Conversation transcript</CardTitle><p className="text-xs text-muted-foreground">Messages appear in order. Customer replies include the score after that turn.</p></CardHeader>
          <CardContent className="space-y-3 p-4 sm:p-5">
            {convo.turns.map((turn, index) => {
              if (index > shownUpTo) return null;
              const point = pointByTurn[turn.turn];
              const delta = index ? convo.scores[index] - convo.scores[index - 1] : 0;
              const person = turn.speaker || (turn.role === 'customer' ? names.customer : names.agent);
              return (
                <motion.div key={turn.turn} ref={(element) => { refs.current[index] = element; }} initial={replay != null ? { opacity: 0, y: 5 } : false} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.18 }}
                  className={`flex scroll-mt-28 gap-2.5 ${turn.role === 'customer' ? 'flex-row-reverse' : ''}`} onMouseEnter={() => setActive(index)}>
                  <span className={`grid h-8 w-8 shrink-0 place-items-center rounded-full text-[11px] font-bold ${turn.role === 'customer' ? 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-100' : 'bg-muted text-muted-foreground'}`} title={person}>{person.trim().charAt(0).toUpperCase()}</span>
                  <div className={`min-w-0 max-w-[88%] rounded-2xl border p-3.5 shadow-sm ${turn.role === 'customer' ? 'border-blue-100 bg-blue-50/70 dark:border-blue-900/50 dark:bg-blue-950/30' : 'border-border bg-card'} ${active === index ? 'ring-1 ring-primary/30' : ''}`}>
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="text-[11px] font-bold">{person}</span><span className="text-[10px] text-muted-foreground">Turn {turn.turn}</span>
                      <Badge variant="outline" className="ml-auto h-5 px-1.5 text-[10px] tabular-nums">{pct(convo.scores[index])}</Badge>
                      {index > 0 && Math.abs(delta) >= 0.01 && <span className={`text-[10px] font-bold ${delta > 0 ? 'text-teal-700 dark:text-teal-200' : 'text-orange-700 dark:text-orange-200'}`}>{pts(delta)}</span>}
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-6">{turn.text}</p>
                    <CueChips cues={turn.cues} />
                    {point && <button type="button" onClick={() => { setPointTurn(point.turn); setShowEvidence(true); }} className={`mt-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-bold ${point.direction === 'rise' ? 'bg-amber-100 text-amber-900 dark:bg-amber-950/60 dark:text-amber-100' : 'bg-sky-100 text-sky-900 dark:bg-sky-950/60 dark:text-sky-100'}`}>{point.direction === 'rise' ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}{point.direction === 'rise' ? 'Rise' : 'Drop'} · {pts(point.delta)} moment</button>}
                  </div>
                </motion.div>
              );
            })}
          </CardContent>
        </Card>

        <Card className="overflow-hidden xl:sticky xl:top-28">
          <CardHeader className="flex-row items-center justify-between border-b border-border/70 pb-4">
            <div><CardTitle>Turning-point evidence</CardTitle><p className="mt-1 text-xs text-muted-foreground">What appears to have shifted the customer's mind?</p></div>
            <Button variant="ghost" size="icon" onClick={() => setShowEvidence((value) => !value)} aria-label={showEvidence ? 'Collapse evidence' : 'Expand evidence'}><ChevronDown size={17} className={`transition-transform ${showEvidence ? 'rotate-180' : ''}`} /></Button>
          </CardHeader>
          {showEvidence && <CardContent className="space-y-4 p-4 sm:p-5">
            {visiblePoints.length === 0 ? <div className="rounded-xl border border-dashed border-border p-5 text-sm text-muted-foreground">{convo.method === 'pelt' ? 'No change points up to this turn.' : 'No turning points up to this turn: the score never moved sharply between two consecutive turns.'}</div> : (
              <>
                <label className="block space-y-2 text-xs font-semibold text-muted-foreground">Select moment
                  <select value={selectedPoint?.turn ?? ''} onChange={(event) => setPointTurn(Number(event.target.value))} className="h-10 w-full rounded-xl border border-input bg-background px-3 text-sm font-medium text-foreground outline-none focus:ring-2 focus:ring-ring">
                    {visiblePoints.map((point) => <option key={point.turn} value={point.turn}>Turn {point.turn} · {point.direction === 'rise' ? 'Rise' : 'Drop'} {pts(point.delta)}</option>)}
                  </select>
                </label>
                {selectedPoint && <EvidenceCard convo={convo} point={selectedPoint} features={topFeatures(selectedPoint.top_features)} names={names} />}
              </>
            )}
            <p className="text-[11px] leading-5 text-muted-foreground">Chips are detected cues: positive raises inclination, negative lowers it, neutral marks an agent move.</p>
          </CardContent>}
        </Card>
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon, secondary }) {
  return <Card><CardContent className="flex items-center gap-3 p-4"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary"><Icon size={18} /></span><div className="min-w-0"><p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p><p className="truncate text-xl font-extrabold">{value}</p>{secondary && <p className="truncate text-[11px] text-muted-foreground">{secondary}</p>}</div></CardContent></Card>;
}

function EvidenceCard({ convo, point, features, names }) {
  const before = convo.turns.find((turn) => turn.turn === point.prev_agent_turn);
  const trigger = convo.turns.find((turn) => turn.turn === point.turn);
  const faithfulnessHelp = 'How much the score changes when the trigger message is removed and the conversation is re-scored. A small change means the explanation is weak.';
  return (
    <div className={`rounded-2xl border p-4 ${point.direction === 'rise' ? 'border-amber-200 bg-amber-50/70 dark:border-amber-900/60 dark:bg-amber-950/20' : 'border-sky-200 bg-sky-50/70 dark:border-sky-900/60 dark:bg-sky-950/20'}`}>
      <p className="text-sm font-bold leading-6">{point.rationale_llm || point.rationale}</p>
      {point.rationale_llm && <p className="mt-1 text-xs leading-5 text-muted-foreground">Written by Gemini. Rule-based version: {point.rationale}</p>}
      <div className="my-3 grid grid-cols-2 gap-2">
        <div className="rounded-xl bg-card/80 p-2.5"><p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Score change</p><b className="mt-1 block tabular-nums">{pts(point.delta)}</b></div>
        <div className="rounded-xl bg-card/80 p-2.5" title={faithfulnessHelp}><p className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Faithfulness <CircleHelp size={11} /></p><b className="mt-1 block">{point.faithfulness == null ? 'n/a' : point.weak ? 'Weak' : 'Strong'}</b></div>
      </div>
      {point.faithfulness != null && <p className="mb-3 text-xs leading-5 text-muted-foreground">Removing the trigger message changes the score by {Math.abs(point.faithfulness * 100).toFixed(0)} pts. A small change means the explanation is weak.</p>}
      <p className="mb-3 text-[10px] font-medium text-muted-foreground">Detected by the {convo.method === 'pelt' ? 'change point' : 'score change'} method.</p>
      {before && <EvidenceMessage title={`Sales message just before · turn ${point.prev_agent_turn}`} turn={before} names={names} />}
      {trigger && <EvidenceMessage title={`Trigger message · turn ${point.turn}`} turn={trigger} names={names} />}
      {features.length > 0 && <div className="mt-3 border-t border-border/60 pt-3"><p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Top model features</p><div className="flex flex-wrap gap-1.5">{features.map((feature) => <Badge key={feature.feature} variant="outline" className="bg-card/80">{feature.feature} ({feature.contribution >= 0 ? '+' : ''}{feature.contribution.toFixed(2)})</Badge>)}</div></div>}
    </div>
  );
}

function EvidenceMessage({ title, turn, names }) {
  return <div className="mt-3 rounded-xl border border-border/70 bg-card/80 p-3"><p className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{title}</p><p className="text-[11px] font-semibold">{turn.speaker || (turn.role === 'customer' ? names.customer : names.agent)}</p><p className="mt-1 text-xs leading-5">{turn.text}</p><CueChips cues={turn.cues} /></div>;
}
