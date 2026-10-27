import { lazy, Suspense, useEffect, useState } from 'react';
import { HashRouter, Link, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { BarChart3, Menu, Moon, Sun, Users, WandSparkles, Workflow, X } from 'lucide-react';
import { DataGate, Mark, SelectionBar, SelectionProvider, useSelection } from './components/ui';
import {
  Button, Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle, DropdownMenu,
  DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, TooltipProvider,
} from './components/ui/primitives';
const Analyze = lazy(() => import('./pages/Analyze'));
const Annotate = lazy(() => import('./pages/Annotate'));
const Conversation = lazy(() => import('./pages/Conversation'));
const Insights = lazy(() => import('./pages/Insights'));
const Leads = lazy(() => import('./pages/Leads'));
const Overview = lazy(() => import('./pages/Overview'));

const NAVIGATION = [
  { to: '/', label: 'Pipeline', icon: Workflow, end: true },
  { to: '/leads', label: 'Leads', icon: Users },
  { to: '/insights', label: 'Insights', icon: BarChart3 },
  { to: '/analyze', label: 'Analyze', icon: WandSparkles },
];

function LiveStatus() {
  const { server } = useSelection();
  const label = server.status === 'ok' ? 'Live data' : server.status === 'loading' ? 'Connecting' : 'Offline';
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs font-semibold text-muted-foreground">
      <span className={`h-2 w-2 rounded-full ${server.status === 'ok' ? 'bg-emerald-500 shadow-[0_0_0_3px_rgb(16_185_129/12%)]' : server.status === 'loading' ? 'animate-pulse bg-amber-500' : 'bg-amber-500'}`} />
      {label}
    </span>
  );
}

function ThemeToggle({ theme, setTheme }) {
  const dark = theme === 'dark';
  const Icon = dark ? Sun : Moon;
  return (
    <Button variant="outline" size="icon" aria-label={`Switch to ${dark ? 'light' : 'dark'} mode`}
      title={`Switch to ${dark ? 'light' : 'dark'} mode`} onClick={() => setTheme(dark ? 'light' : 'dark')}
      className="h-10 w-10 rounded-full bg-card/80">
      <Icon size={17} />
    </Button>
  );
}

function TopBar({ theme, setTheme }) {
  const location = useLocation();
  return (
    <>
      <header className="sticky top-0 z-40 border-b border-border/80 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto flex min-h-[72px] max-w-[1500px] items-center gap-5 px-5 lg:px-8">
          <Link to="/" className="flex shrink-0 items-center gap-2.5 text-foreground" aria-label="Inclination home">
            <Mark size={34} />
            <span className="text-[17px] font-extrabold tracking-[-0.04em]">Inclination</span>
          </Link>
          <nav className="hidden min-w-0 flex-1 items-center gap-1 lg:flex" aria-label="Main navigation">
            {NAVIGATION.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => (
                `inline-flex h-10 items-center gap-2 rounded-xl px-3.5 text-sm font-semibold transition-colors ${isActive ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`
              )}>
                <Icon size={16} strokeWidth={1.9} /><span>{label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex shrink-0 items-center gap-2.5">
            <div className="hidden xl:block"><SelectionBar /></div>
            <LiveStatus />
            <ThemeToggle theme={theme} setTheme={setTheme} />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="Team tools menu" className="h-10 w-10 rounded-full bg-card/80">
                  <Menu size={17} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <div className="px-2.5 py-2 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">Team tools</div>
                <DropdownMenuItem asChild>
                  <Link to="/annotate"><WandSparkles size={16} />Annotate</Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <div className="border-t border-border/70 px-5 py-2 xl:hidden"><SelectionBar /></div>
        <div className="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-background/95 px-2 pb-[max(env(safe-area-inset-bottom),8px)] pt-2 backdrop-blur-lg lg:hidden">
          <nav aria-label="Mobile navigation" className="mx-auto grid max-w-lg grid-cols-4 gap-1">
            {NAVIGATION.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => (
                `flex min-h-12 flex-col items-center justify-center gap-1 rounded-xl text-[10px] font-bold ${isActive || (location.pathname.startsWith('/conversations') && to === '/leads') ? 'text-primary' : 'text-muted-foreground'}`
              )}>
                <Icon size={18} /><span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
    </>
  );
}

function ConversationDrawer({ onClose }) {
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="left-auto right-0 top-0 h-[100dvh] w-full max-w-[1100px] translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none border-l p-0 sm:rounded-l-3xl">
        <div className="sr-only">
          <DialogTitle>Conversation detail</DialogTitle>
          <DialogDescription>Score, turning points and transcript for this lead.</DialogDescription>
        </div>
        <DialogClose asChild><Button variant="outline" size="icon" aria-label="Close conversation" className="absolute right-4 top-4 z-30 rounded-full bg-card/90"><X size={17} /></Button></DialogClose>
        <div className="h-full overflow-y-auto">
          <Suspense fallback={<PageFallback />}><Conversation /></Suspense>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function AppLayout({ theme, setTheme }) {
  const location = useLocation();
  const navigate = useNavigate();
  const background = location.state?.backgroundLocation;
  const isConversation = location.pathname.startsWith('/conversations');
  const baseLocation = background ?? (isConversation ? { ...location, pathname: '/leads' } : location);
  const closeConversation = () => {
    if (background) navigate(-1);
    else navigate('/leads', { replace: true });
  };

  return (
    <TooltipProvider delayDuration={250}>
      <div className="min-h-screen bg-background pb-20 text-foreground lg:pb-0">
        <TopBar theme={theme} setTheme={setTheme} />
        <div className="mx-auto min-h-[calc(100vh-72px)] max-w-[1500px]">
          <Routes location={baseLocation}>
            <Route path="/" element={<Suspense fallback={<PageFallback />}><DataGate><Overview /></DataGate></Suspense>} />
            <Route path="/leads" element={<Suspense fallback={<PageFallback />}><DataGate><Leads /></DataGate></Suspense>} />
            <Route path="/insights" element={<Suspense fallback={<PageFallback />}><DataGate><Insights /></DataGate></Suspense>} />
            <Route path="/annotate" element={<Suspense fallback={<PageFallback />}><Annotate /></Suspense>} />
            <Route path="/analyze" element={<Suspense fallback={<PageFallback />}><Analyze /></Suspense>} />
            <Route path="/conversations/*" element={<Suspense fallback={<PageFallback />}><DataGate><Leads /></DataGate></Suspense>} />
            <Route path="*" element={<Suspense fallback={<PageFallback />}><DataGate><Overview /></DataGate></Suspense>} />
          </Routes>
        </div>
        {isConversation && <Routes location={location}><Route path="/conversations/:id" element={<ConversationDrawer onClose={closeConversation} />} /><Route path="/conversations/*" element={<ConversationDrawer onClose={closeConversation} />} /></Routes>}
      </div>
    </TooltipProvider>
  );
}

function PageFallback() {
  return <main className="mx-auto grid max-w-7xl gap-5 px-5 py-8" aria-live="polite"><div className="h-10 w-64 animate-pulse rounded-lg bg-muted" /><div className="h-44 animate-pulse rounded-2xl bg-muted" /><div className="grid gap-4 md:grid-cols-3"><div className="h-36 animate-pulse rounded-2xl bg-muted" /><div className="h-36 animate-pulse rounded-2xl bg-muted" /><div className="h-36 animate-pulse rounded-2xl bg-muted" /></div></main>;
}

function ThemeRoot() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('inclination-theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch { /* use the system preference */ }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try { localStorage.setItem('inclination-theme', theme); } catch { /* storage unavailable */ }
  }, [theme]);
  return <AppLayout theme={theme} setTheme={setTheme} />;
}

export default function App() {
  return <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><SelectionProvider><ThemeRoot /></SelectionProvider></HashRouter>;
}
