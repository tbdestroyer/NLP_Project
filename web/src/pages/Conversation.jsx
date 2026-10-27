import { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { ChevronDown, MoreHorizontal, RotateCcw, SlidersHorizontal } from 'lucide-react';
import { getConversation, getLeads } from '../api/client';
import ConversationDetail from '../components/ConversationDetail';
import { ErrorNote, pct, Segmented, SyntheticNote, useAsync, useSelection } from '../components/ui';
import { MODELS } from '../data/taxonomy';
import {
  Badge, Button, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Skeleton,
} from '../components/ui/primitives';

const METHODS = [{ value: 'delta', label: 'Score change' }, { value: 'pelt', label: 'Change point' }];
const TAGGERS = [{ value: 'rules', label: 'Rules' }, { value: 'llm', label: 'Gemini' }];

export default function Conversation() {
  const { id } = useParams();
  const selection = useSelection();
  const location = useLocation();
  const navigate = useNavigate();
  const [method, setMethod] = useState('delta');
  const [tagger, setTagger] = useState('rules');
  const [overlays, setOverlays] = useState([]);
  const modelOptions = selection.server.options[selection.dataset] ?? [];
  const availableOverlays = modelOptions.filter((model) => model !== selection.model);
  const leadState = useAsync(() => getLeads(selection), [selection.dataset, selection.model, selection.split]);
  const conversationState = useAsync(
    () => id ? getConversation(id, selection.model, { method, tagger, models: overlays }) : null,
    [id, selection.model, method, tagger, overlays.join(',')],
  );

  useEffect(() => {
    if (!id && leadState.data?.length) {
      const first = [...leadState.data].sort((a, b) => b.final_score - a.final_score)[0];
      navigate(`/conversations/${encodeURIComponent(first.conversation_id)}`, { replace: true });
    }
  }, [id, leadState.data, navigate]);
  useEffect(() => { setMethod('delta'); setTagger('rules'); setOverlays([]); }, [id, selection.model]);
  useEffect(() => {
    const convo = conversationState.data;
    if (!convo) return;
    const patch = {};
    if (convo.dataset !== selection.dataset) patch.dataset = convo.dataset;
    if (selection.split !== 'all' && convo.split !== selection.split) patch.split = convo.split;
    if (Object.keys(patch).length) selection.set(patch);
  }, [conversationState.data, selection.dataset, selection.split, selection.set]);

  if (leadState.error) return <ErrorNote error={leadState.error} />;
  if (conversationState.error) return <ErrorNote error={conversationState.error} />;
  if (!leadState.data || !conversationState.data) return <ConversationSkeleton />;
  if (!leadState.data.length) return <div className="p-8 text-center text-sm text-muted-foreground">No conversations in this selection. Pick another dataset or split.</div>;

  const leads = [...leadState.data].sort((a, b) => b.final_score - a.final_score);
  const currentLead = leads.find((lead) => lead.conversation_id === id);
  if (!currentLead) return (
    <div className="p-8 text-center"><h2 className="text-lg font-bold">Conversation not in this selection</h2><p className="mt-2 text-sm text-muted-foreground">Choose a conversation from the current dataset, model, and split.</p><Button variant="outline" className="mt-4" onClick={() => navigate('/leads')}>Open leads</Button></div>
  );

  const convo = conversationState.data;
  if (convo.dataset !== selection.dataset || (selection.split !== 'all' && convo.split !== selection.split)) {
    return <p className="p-8 text-sm text-muted-foreground" aria-live="polite">Opening conversation from its dataset and split…</p>;
  }
  const hasMethodSwitch = convo.available_methods.includes('pelt');
  const hasTaggerSwitch = convo.available_taggers.includes('llm');
  const overlayModel = (model) => setOverlays((current) => current.includes(model) ? current.filter((item) => item !== model) : [...current, model]);

  return (
    <div className="min-h-full bg-background">
      <div className="sticky top-0 z-20 border-b border-border bg-background/95 px-5 py-4 backdrop-blur-xl md:px-8">
        <div className="flex flex-wrap items-center justify-between gap-3 pr-12 sm:pr-0">
          <div className="min-w-0 flex-1">
            <p className="mb-1 text-[10px] font-bold uppercase tracking-[.16em] text-primary">Conversation workspace</p>
            <Select value={id} onValueChange={(value) => { setMethod('delta'); navigate(`/conversations/${encodeURIComponent(value)}`, { state: location.state }); }}>
              <SelectTrigger aria-label="Choose conversation" className="h-auto max-w-full border-0 bg-transparent p-0 text-base font-bold shadow-none focus:ring-0 md:text-lg"><SelectValue /></SelectTrigger>
              <SelectContent className="max-w-[min(90vw,680px)]">{leads.map((lead) => <SelectItem key={lead.conversation_id} value={lead.conversation_id}>{pct(lead.final_score)} · {lead.title}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="hidden bg-card sm:inline-flex">{MODELS[selection.model]?.label ?? selection.model}</Badge>
            <DropdownMenu>
              <DropdownMenuTrigger asChild><Button variant="outline" size="sm" className="rounded-xl"><MoreHorizontal size={16} />More</Button></DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-64">
                <div className="px-2.5 py-2 text-xs font-bold">Overlay other models</div>
                {availableOverlays.length ? availableOverlays.map((model) => (
                  <DropdownMenuItem key={model} onSelect={(event) => { event.preventDefault(); overlayModel(model); }} className="justify-between">
                    <span>{MODELS[model]?.label ?? model}</span><input type="checkbox" readOnly checked={overlays.includes(model)} className="h-4 w-4 accent-primary" />
                  </DropdownMenuItem>
                )) : <div className="px-2.5 pb-2 text-xs text-muted-foreground">No other models are available for this dataset.</div>}
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => { setMethod('delta'); setTagger('rules'); setOverlays([]); }}><RotateCcw size={15} />Reset controls</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        {(hasMethodSwitch || hasTaggerSwitch) && (
          <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-border/70 pt-3">
            {hasMethodSwitch && <ControlGroup title="Turning points" help="Score change finds sharp one-turn jumps; change point looks for lasting shifts in the curve."><Segmented label="Turning-point method" value={method} onChange={setMethod} options={METHODS.filter((option) => convo.available_methods.includes(option.value))} /></ControlGroup>}
            {hasTaggerSwitch && <ControlGroup title="Cue source" help="Rules are keyword patterns. Gemini tags are available for conversations with LLM cue output."><Segmented label="Cue source" value={tagger} onChange={setTagger} options={TAGGERS.filter((option) => convo.available_taggers.includes(option.value))} /></ControlGroup>}
          </div>
        )}
        {availableOverlays.length > 0 && overlays.length > 0 && (
          <p className="mt-2 text-[11px] text-muted-foreground">Overlays: {overlays.map((model) => MODELS[model]?.short ?? model).join(', ')}</p>
        )}
      </div>
      <SyntheticNote dataset={convo.dataset} />
      <ConversationDetail key={`${id}-${selection.model}-${method}-${tagger}`} convo={convo}
        showOutcome={selection.evalMode} model={selection.model} split={selection.split} />
    </div>
  );
}

function ControlGroup({ title, help, children }) {
  return <div className="flex flex-wrap items-center gap-2" title={help}><span className="text-[11px] font-semibold text-muted-foreground">{title}</span>{children}</div>;
}

function ConversationSkeleton() {
  return <div className="space-y-5 p-6"><Skeleton className="h-12" /><Skeleton className="h-44" /><div className="grid gap-5 lg:grid-cols-2"><Skeleton className="h-80" /><Skeleton className="h-80" /></div></div>;
}
