import { createContext, lazy, Suspense, useContext, useEffect, useMemo, useState } from 'react';
import { Flame, Snowflake, TrendingUp, Minus, CircleHelp, CheckCircle2 } from 'lucide-react';
import { getHealth, getOptions, getUIConfig } from '../api/client';
import { DATASETS, MODELS } from '../data/taxonomy';
import {
  Badge, Button, Card, CardContent, Popover, PopoverContent, PopoverTrigger,
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Toggle,
} from './ui/primitives';

const RechartsSparkline = lazy(() => import('./Sparkline'));
const SelectionContext = createContext(null);
const KEY = 'inclination-selection';

function readStored() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}

export function SelectionProvider({ children }) {
  const [selection, setSelection] = useState(() => ({
    dataset: 'craigslist', model: 'v1', split: 'test', evalMode: false, ...readStored(),
  }));
  const [server, setServer] = useState({ status: 'loading', options: {}, health: null, config: null });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const health = await getHealth();
        if (!health.ok) {
          if (alive) setServer({ status: 'missing', options: {}, health });
          return;
        }
        const [options, config] = await Promise.all([getOptions(), getUIConfig()]);
        if (alive) setServer({ status: 'ok', options, health, config });
      } catch {
        if (alive) setServer({ status: 'down', options: {}, health: null, config: null });
      }
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const datasets = Object.keys(server.options);
    if (!datasets.length) return;
    setSelection((current) => {
      const dataset = datasets.includes(current.dataset) ? current.dataset : datasets[0];
      const models = server.options[dataset];
      const model = models.includes(current.model) ? current.model : models[0];
      return dataset === current.dataset && model === current.model ? current : { ...current, dataset, model };
    });
  }, [server.options, selection.dataset]);

  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(selection)); } catch { /* storage unavailable */ }
  }, [selection]);

  const value = useMemo(() => ({
    ...selection,
    server,
    set: (patch) => setSelection((current) => ({ ...current, ...patch })),
  }), [selection, server]);
  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>;
}

export const useSelection = () => useContext(SelectionContext);

export function useAsync(fn, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    setState((current) => ({ ...current, loading: true }));
    Promise.resolve(fn())
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function Segmented({ options, value, onChange, label, className = '' }) {
  return (
    <div className={`inline-flex items-center gap-1 rounded-xl bg-muted p-1 ${className}`} role="group" aria-label={label}>
      {options.map((option) => (
        <Toggle key={option.value} pressed={option.value === value} disabled={option.disabled}
          title={option.title} onPressedChange={() => onChange(option.value)}
          className="h-8 rounded-lg px-3 text-xs font-semibold text-muted-foreground data-[state=on]:bg-card data-[state=on]:text-foreground data-[state=on]:shadow-sm">
          {option.label}
        </Toggle>
      ))}
    </div>
  );
}

export function SelectionBar() {
  const selection = useSelection();
  const available = selection.server.options;
  const datasets = Object.keys(available).length ? Object.keys(available) : Object.keys(DATASETS);
  const models = (available[selection.dataset] ?? []).filter((key) => key in MODELS);
  const splitOptions = [
    { value: 'test', label: 'Test' }, { value: 'val', label: 'Validation' },
    { value: 'train', label: 'Train' }, { value: 'all', label: 'All' },
  ];
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" className="h-10 rounded-full border-border/80 bg-card/80 px-4 text-xs font-semibold shadow-sm">
          <span className="max-w-32 truncate">{DATASETS[selection.dataset]?.displayLabel ?? DATASETS[selection.dataset]?.label ?? selection.dataset}</span>
          <span className="text-muted-foreground">·</span>
          <span>{MODELS[selection.model]?.short ?? selection.model}</span>
          <span className="text-muted-foreground">·</span>
          <span className="capitalize">{selection.split === 'val' ? 'Validation' : selection.split}</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(92vw,370px)] space-y-4">
        <div>
          <p className="mb-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">Dataset</p>
          <Select value={selection.dataset} onValueChange={(dataset) => selection.set({ dataset })}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>{datasets.map((dataset) => (
              <SelectItem key={dataset} value={dataset}>{DATASETS[dataset]?.displayLabel ?? DATASETS[dataset]?.label ?? dataset}</SelectItem>
            ))}</SelectContent>
          </Select>
        </div>
        {models.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">Model</p>
            <Segmented label="Model" value={selection.model} onChange={(model) => selection.set({ model })}
              options={models.map((key) => ({ value: key, label: MODELS[key]?.label ?? key }))} />
          </div>
        )}
        <div title="The test split is the honest view: the model never saw it.">
          <p className="mb-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">Split</p>
          <Segmented label="Split" value={selection.split} onChange={(split) => selection.set({ split })} options={splitOptions} />
        </div>
        <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-border p-3 text-sm">
          <input type="checkbox" className="h-4 w-4 accent-primary" checked={selection.evalMode}
            onChange={(event) => selection.set({ evalMode: event.target.checked })} />
          <span><b className="block text-xs">Show true outcomes</b><small className="text-muted-foreground">Evaluation mode</small></span>
        </label>
      </PopoverContent>
    </Popover>
  );
}

export function DataGate({ children }) {
  const { server } = useSelection();
  if (server.status === 'ok') return children;
  if (server.status === 'loading') {
    return (
      <div className="mx-auto grid max-w-5xl gap-5 p-6 pt-12" aria-live="polite">
        <div className="h-9 w-64 animate-pulse rounded-lg bg-muted" />
        <div className="h-44 animate-pulse rounded-2xl bg-muted" />
        <div className="grid gap-4 md:grid-cols-3">{[0, 1, 2].map((item) => <div key={item} className="h-36 animate-pulse rounded-2xl bg-muted" />)}</div>
      </div>
    );
  }
  const down = server.status === 'down';
  return (
    <div className="mx-auto max-w-3xl px-5 py-16">
      <Card className="overflow-hidden">
        <div className="bg-gradient-to-r from-navy via-blue-700 to-sky-500 px-7 py-8 text-white">
          <div className="mb-3 text-xs font-bold uppercase tracking-[.18em] text-blue-100">Connection needed</div>
          <h1 className="text-3xl font-extrabold tracking-tight">{down ? 'Start the data server' : 'Pipeline outputs are missing'}</h1>
          <p className="mt-3 max-w-xl text-sm leading-6 text-blue-50">
            {down
              ? "These screens read the real pipeline results through a small Python server, and it isn't running."
              : `The data server is running but can't find ${server.health.missing.join(', ')} in ${server.health.outputs}.`}
          </p>
        </div>
        <CardContent className="space-y-4 pt-6 text-sm">
          {down ? (
            <>
              <p>In a second terminal, from the repository:</p>
              <code className="block rounded-xl bg-muted p-4 font-mono text-sm">cd inclination_score<br />make api</code>
              <p className="text-muted-foreground">Keep the server terminal open, then reload this page.</p>
            </>
          ) : (
            <>
              <p>Place the generated exports in <code>inclination_score/outputs/</code>.</p>
              <code className="block rounded-xl bg-muted p-4 font-mono text-sm">cd inclination_score<br />make all<br />make api</code>
              <p className="text-muted-foreground">The app reads these results; it does not train models.</p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export function ErrorNote({ error }) {
  return (
    <div role="alert" className="mx-auto max-w-4xl px-5 py-8">
      <Card className="border-rose-200 bg-rose-50 p-5 text-rose-900 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-100">
        <b>Couldn't load this from the data server.</b>
        <p className="mt-1 text-sm">{error.message} Check the terminal running <code>make api</code> for details.</p>
      </Card>
    </div>
  );
}

export const cap = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
export const pct = (value, digits = 0) => (value == null ? 'n/a' : `${(value * 100).toFixed(digits)}%`);
export const pts = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(0)} pts`;

const STATUS_DETAILS = {
  Hot: { Icon: Flame, color: 'border-orange-200 bg-orange-50 text-orange-800 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-200' },
  Rising: { Icon: TrendingUp, color: 'border-teal-200 bg-teal-50 text-teal-800 dark:border-teal-900 dark:bg-teal-950/40 dark:text-teal-200' },
  Stalled: { Icon: Minus, color: 'border-slate-200 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200' },
  Cooling: { Icon: Snowflake, color: 'border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-200' },
};

export function StatusTag({ status, className = '' }) {
  const detail = STATUS_DETAILS[status] ?? STATUS_DETAILS.Stalled;
  const { Icon } = detail;
  return <Badge variant="outline" className={`whitespace-nowrap ${detail.color} ${className}`}><Icon size={13} aria-hidden="true" />{status}</Badge>;
}

export const STATUS_COLOR = { Hot: '#f97316', Rising: '#0f9f88', Stalled: '#64748b', Cooling: '#0284c7' };

export function Sparkline({ values, width = 84, height = 26, className = '' }) {
  if (!values?.length) return null;
  return (
    <Suspense fallback={<span className={`inline-block shrink-0 ${className}`} style={{ width, height }} aria-hidden="true" />}>
      <RechartsSparkline values={values} width={width} height={height} className={className} />
    </Suspense>
  );
}

export function Icon({ name, size = 18 }) {
  const icons = { help: CircleHelp, check: CheckCircle2, flame: Flame, snow: Snowflake, trend: TrendingUp, minus: Minus };
  const IconComponent = icons[name] ?? CircleHelp;
  return <IconComponent size={size} aria-hidden="true" />;
}

export function EmptyIllustration() {
  return (
    <div className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary">
      <CheckCircle2 size={26} strokeWidth={1.8} aria-hidden="true" />
    </div>
  );
}

export function Mark({ size = 32 }) {
  return (
    <span style={{ width: size, height: size }} className="grid shrink-0 place-items-center rounded-xl bg-gradient-to-br from-blue-700 via-blue-600 to-sky-400 text-white shadow-sm">
      <svg width={size * 0.65} height={size * 0.65} viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M3 18h5l3-3 3-8 3 3h4" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="14" cy="7" r="2" fill="currentColor" />
      </svg>
    </span>
  );
}

export function SyntheticNote({ dataset }) {
  if (!DATASETS[dataset]?.synthetic) return null;
  return <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100"><b>Synthetic data.</b> These SaaS chats were generated with GPT-4. Use them to check the pipeline works, not as evidence.</div>;
}
