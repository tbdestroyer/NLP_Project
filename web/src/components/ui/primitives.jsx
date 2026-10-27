import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import * as SelectPrimitive from '@radix-ui/react-select';
import * as SliderPrimitive from '@radix-ui/react-slider';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import * as TogglePrimitive from '@radix-ui/react-toggle';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { Check, ChevronDown, ChevronUp, X } from 'lucide-react';
import { Slot } from '@radix-ui/react-slot';
import { cn } from '@/lib/utils';

const buttonStyles = {
  default: 'bg-primary text-primary-foreground shadow-sm hover:bg-primary/90',
  secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
  outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
  ghost: 'hover:bg-accent hover:text-accent-foreground',
  destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
  link: 'text-primary underline-offset-4 hover:underline',
};

export const Button = React.forwardRef(function Button(
  { className, variant = 'default', size = 'default', asChild = false, ...props }, ref,
) {
  const Component = asChild ? Slot : 'button';
  const sizes = {
    default: 'h-10 px-4 py-2',
    sm: 'h-9 rounded-lg px-3 text-xs',
    lg: 'h-11 rounded-xl px-8',
    icon: 'h-10 w-10',
  };
  return <Component ref={ref} className={cn(
    'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]',
    buttonStyles[variant], sizes[size], className,
  )} {...props} />;
});

export function Card({ className, ...props }) {
  return <div className={cn('rounded-2xl border border-border bg-card text-card-foreground shadow-soft', className)} {...props} />;
}
export function CardHeader({ className, ...props }) {
  return <div className={cn('flex flex-col space-y-1.5 p-6', className)} {...props} />;
}
export function CardTitle({ className, ...props }) {
  return <h3 className={cn('text-base font-bold tracking-tight', className)} {...props} />;
}
export function CardDescription({ className, ...props }) {
  return <p className={cn('text-sm leading-6 text-muted-foreground', className)} {...props} />;
}
export function CardContent({ className, ...props }) {
  return <div className={cn('p-6 pt-0', className)} {...props} />;
}
export function CardFooter({ className, ...props }) {
  return <div className={cn('flex items-center p-6 pt-0', className)} {...props} />;
}

export const Badge = React.forwardRef(function Badge({ className, variant = 'default', ...props }, ref) {
  const variants = {
    default: 'border-transparent bg-primary text-primary-foreground',
    secondary: 'border-transparent bg-secondary text-secondary-foreground',
    outline: 'border-border text-foreground',
    destructive: 'border-transparent bg-destructive text-destructive-foreground',
  };
  return <span ref={ref} className={cn(
    'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-semibold transition-colors',
    variants[variant], className,
  )} {...props} />;
});

export const Tabs = TabsPrimitive.Root;
export const TabsList = React.forwardRef(function TabsList({ className, ...props }, ref) {
  return <TabsPrimitive.List ref={ref} className={cn(
    'inline-flex h-10 items-center justify-center rounded-xl bg-muted p-1 text-muted-foreground',
    className,
  )} {...props} />;
});
export const TabsTrigger = React.forwardRef(function TabsTrigger({ className, ...props }, ref) {
  return <TabsPrimitive.Trigger ref={ref} className={cn(
    'inline-flex items-center justify-center whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm',
    className,
  )} {...props} />;
});
export const TabsContent = React.forwardRef(function TabsContent({ className, ...props }, ref) {
  return <TabsPrimitive.Content ref={ref} className={cn(
    'mt-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    className,
  )} {...props} />;
});

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;
export const DialogTitle = DialogPrimitive.Title;
export const DialogDescription = DialogPrimitive.Description;
export const DialogPortal = DialogPrimitive.Portal;
export const DialogOverlay = React.forwardRef(function DialogOverlay({ className, ...props }, ref) {
  return <DialogPrimitive.Overlay ref={ref} className={cn(
    'fixed inset-0 z-50 bg-slate-950/45 backdrop-blur-[2px] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
    className,
  )} {...props} />;
});
export const DialogContent = React.forwardRef(function DialogContent({ className, children, ...props }, ref) {
  return (
    <DialogPortal>
      <DialogOverlay />
      <DialogPrimitive.Content ref={ref} className={cn(
        'fixed left-1/2 top-1/2 z-50 grid w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 -translate-y-1/2 gap-4 rounded-2xl border border-border bg-card p-6 shadow-float duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
        className,
      )} {...props}>{children}</DialogPrimitive.Content>
    </DialogPortal>
  );
});
export const DialogHeader = ({ className, ...props }) => <div className={cn('flex flex-col gap-1.5 text-left', className)} {...props} />;
export const DialogFooter = ({ className, ...props }) => <div className={cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)} {...props} />;

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;
export function SheetContent({ side = 'right', className, children, ...props }) {
  const sides = {
    right: 'inset-y-0 right-0 h-full w-[min(100vw,560px)] border-l data-[state=open]:slide-in-from-right data-[state=closed]:slide-out-to-right',
    left: 'inset-y-0 left-0 h-full w-[min(100vw,560px)] border-r data-[state=open]:slide-in-from-left data-[state=closed]:slide-out-to-left',
    top: 'inset-x-0 top-0 max-h-[85vh] w-full border-b data-[state=open]:slide-in-from-top data-[state=closed]:slide-out-to-top',
    bottom: 'inset-x-0 bottom-0 max-h-[85vh] w-full border-t data-[state=open]:slide-in-from-bottom data-[state=closed]:slide-out-to-bottom',
  };
  return (
    <DialogPortal>
      <DialogOverlay />
      <DialogPrimitive.Content className={cn(
        'fixed z-50 flex flex-col gap-4 border-border bg-card p-6 shadow-float duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out',
        sides[side], className,
      )} {...props}>
      {children}
      <DialogPrimitive.Close aria-label="Close panel" className="absolute right-4 top-4 grid h-9 w-9 place-items-center rounded-full border border-border bg-card text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"><X size={16} /></DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPortal>
  );
}
export const SheetHeader = DialogHeader;
export const SheetTitle = DialogTitle;
export const SheetDescription = DialogDescription;

export const DropdownMenu = DropdownMenuPrimitive.Root;
export const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger;
export const DropdownMenuGroup = DropdownMenuPrimitive.Group;
export const DropdownMenuItem = React.forwardRef(function DropdownMenuItem({ className, inset, ...props }, ref) {
  return <DropdownMenuPrimitive.Item ref={ref} className={cn(
    'relative flex cursor-default select-none items-center gap-2 rounded-lg px-2.5 py-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    inset && 'pl-8', className,
  )} {...props} />;
});
export const DropdownMenuContent = React.forwardRef(function DropdownMenuContent({ className, sideOffset = 6, ...props }, ref) {
  return <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content ref={ref} sideOffset={sideOffset} className={cn(
      'z-50 min-w-48 overflow-hidden rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-float data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className,
    )} {...props} />
  </DropdownMenuPrimitive.Portal>;
});
export const DropdownMenuSeparator = (props) => <DropdownMenuPrimitive.Separator className="mx-1 my-1 h-px bg-border" {...props} />;

export const Popover = PopoverPrimitive.Root;
export const PopoverTrigger = PopoverPrimitive.Trigger;
export const PopoverContent = React.forwardRef(function PopoverContent({ className, sideOffset = 8, ...props }, ref) {
  return <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content ref={ref} sideOffset={sideOffset} className={cn(
      'z-50 w-80 rounded-2xl border border-border bg-popover p-4 text-popover-foreground shadow-float outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
      className,
    )} {...props} />
  </PopoverPrimitive.Portal>;
});

export const Select = SelectPrimitive.Root;
export const SelectGroup = SelectPrimitive.Group;
export const SelectValue = SelectPrimitive.Value;
export const SelectTrigger = React.forwardRef(function SelectTrigger({ className, children, ...props }, ref) {
  return <SelectPrimitive.Trigger ref={ref} className={cn(
    'flex h-10 w-full items-center justify-between gap-2 rounded-xl border border-input bg-background px-3 text-sm shadow-sm outline-none focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50 [&>span]:truncate',
    className,
  )} {...props}>
    {children}<SelectPrimitive.Icon asChild><ChevronDown className="h-4 w-4 opacity-60" /></SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>;
});
export const SelectContent = React.forwardRef(function SelectContent({ className, children, position = 'popper', ...props }, ref) {
  return <SelectPrimitive.Portal>
    <SelectPrimitive.Content ref={ref} position={position} className={cn(
      'relative z-50 max-h-96 min-w-32 overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-float data-[state=open]:animate-in data-[state=closed]:animate-out',
      className,
    )} {...props}>
      <SelectPrimitive.ScrollUpButton className="flex cursor-default items-center justify-center py-1"><ChevronUp className="h-4 w-4" /></SelectPrimitive.ScrollUpButton>
      <SelectPrimitive.Viewport className="p-1.5">{children}</SelectPrimitive.Viewport>
      <SelectPrimitive.ScrollDownButton className="flex cursor-default items-center justify-center py-1"><ChevronDown className="h-4 w-4" /></SelectPrimitive.ScrollDownButton>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>;
});
export const SelectItem = React.forwardRef(function SelectItem({ className, children, ...props }, ref) {
  return <SelectPrimitive.Item ref={ref} className={cn(
    'relative flex w-full cursor-default select-none items-center rounded-lg py-2 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    className,
  )} {...props}>
    <span className="absolute left-2 flex h-4 w-4 items-center justify-center"><SelectPrimitive.ItemIndicator><Check className="h-4 w-4" /></SelectPrimitive.ItemIndicator></span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>;
});

export const TooltipProvider = TooltipPrimitive.Provider;
export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;
export const TooltipContent = React.forwardRef(function TooltipContent({ className, sideOffset = 5, ...props }, ref) {
  return <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content ref={ref} sideOffset={sideOffset} className={cn(
      'z-50 overflow-hidden rounded-lg bg-navy px-3 py-1.5 text-xs text-white shadow-float animate-in fade-in-0 zoom-in-95',
      className,
    )} {...props}><TooltipPrimitive.Arrow className="fill-navy" /></TooltipPrimitive.Content>
  </TooltipPrimitive.Portal>;
});

export const Slider = React.forwardRef(function Slider({ className, ...props }, ref) {
  return <SliderPrimitive.Root ref={ref} className={cn(
    'relative flex w-full touch-none select-none items-center py-2',
    className,
  )} {...props}>
    <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
      <SliderPrimitive.Range className="absolute h-full bg-primary" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background shadow-md ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>;
});

export const Toggle = React.forwardRef(function Toggle({ className, ...props }, ref) {
  return <TogglePrimitive.Root ref={ref} className={cn(
    'inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground',
    className,
  )} {...props} />;
});

export function Skeleton({ className, ...props }) {
  return <div className={cn('animate-pulse rounded-xl bg-muted', className)} {...props} />;
}

export function Table({ className, ...props }) {
  return <table className={cn('w-full caption-bottom text-sm', className)} {...props} />;
}
export function TableHeader({ className, ...props }) {
  return <thead className={cn('[&_tr]:border-b [&_tr]:border-border', className)} {...props} />;
}
export function TableBody({ className, ...props }) {
  return <tbody className={cn('[&_tr:last-child]:border-0', className)} {...props} />;
}
export function TableFooter({ className, ...props }) {
  return <tfoot className={cn('border-t border-border bg-muted/50 font-medium [&>tr]:last:border-b-0', className)} {...props} />;
}
export function TableRow({ className, ...props }) {
  return <tr className={cn('border-b border-border transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted', className)} {...props} />;
}
export function TableHead({ className, ...props }) {
  return <th className={cn('h-11 px-4 text-left align-middle text-xs font-semibold text-muted-foreground [&:has([role=checkbox])]:pr-0', className)} {...props} />;
}
export function TableCell({ className, ...props }) {
  return <td className={cn('p-4 align-middle [&:has([role=checkbox])]:pr-0', className)} {...props} />;
}
export function TableCaption({ className, ...props }) {
  return <caption className={cn('mt-4 text-sm text-muted-foreground', className)} {...props} />;
}
