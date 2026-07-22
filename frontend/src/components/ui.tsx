"use client";

import { ReactNode } from "react";

type PanelTone = "default" | "dark" | "amber";

/**
 * Surface card. `tone` exists so a page can anchor the eye: a wall of
 * identical white cards on cream reads as one flat plane, so explainer
 * blocks go dark and control bars go amber.
 */
export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
  tone = "default",
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  tone?: PanelTone;
}) {
  const shell = {
    default: "bg-[var(--panel)] border-[var(--line)]",
    dark: "bg-[var(--dark)] border-[var(--dark)] text-[#f6f1e8]",
    amber: "bg-[var(--amber-tint)] border-[var(--amber-line)]",
  }[tone];

  const titleColor = tone === "dark" ? "text-white" : "text-[var(--text)]";
  const subColor = tone === "dark" ? "text-white/55" : "text-[var(--muted)]";
  const divider = tone === "dark" ? "border-white/12" : "border-[var(--line)]";

  return (
    <div
      className={`rounded-[var(--radius)] border ${shell} ${className}`}
      style={{ boxShadow: tone === "dark" ? "var(--shadow)" : "var(--shadow-sm)" }}
    >
      {(title || right) && (
        <div className={`flex items-start justify-between gap-4 border-b ${divider} px-5 py-3.5`}>
          <div>
            {title && <h3 className={`text-[14px] font-semibold ${titleColor}`}>{title}</h3>}
            {subtitle && (
              <p className={`mt-1 text-[12px] leading-snug ${subColor}`}>{subtitle}</p>
            )}
          </div>
          {right}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

const TONES = {
  default: { fg: "var(--text)", bg: "var(--panel-2)" },
  good: { fg: "var(--good)", bg: "var(--good-wash)" },
  bad: { fg: "var(--bad)", bg: "var(--bad-wash)" },
  warn: { fg: "var(--warn)", bg: "var(--warn-wash)" },
  accent: { fg: "var(--accent)", bg: "var(--accent-wash)" },
  violet: { fg: "var(--violet)", bg: "var(--violet-wash)" },
} as const;

/**
 * Figure block. The value sits on a tinted plate with a colour rail, so a row
 * of these reads as data rather than as four empty white rectangles — and an
 * idle "—" still looks deliberate instead of broken.
 */
export function Stat({
  label,
  value,
  unit,
  tone = "default",
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: keyof typeof TONES;
  hint?: string;
}) {
  const { fg, bg } = TONES[tone];
  const idle = value === "—" || value === "" || value === undefined;
  return (
    <div
      className="relative overflow-hidden rounded-[var(--radius)] border border-[var(--line)] bg-[var(--panel)] pl-4 pr-4 py-3.5"
      style={{ boxShadow: "var(--shadow-sm)" }}
    >
      {/* colour rail */}
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: idle ? "var(--line)" : fg }}
        aria-hidden
      />
      <div className="text-[10.5px] font-semibold uppercase tracking-[0.1em] text-[var(--dim)]">
        {label}
      </div>
      <div
        className="mono mt-2 inline-flex items-baseline gap-1 rounded-lg px-2 py-1"
        style={{ background: idle ? "var(--panel-2)" : bg }}
      >
        <span
          className="text-[24px] font-bold leading-none tabular-nums"
          style={{ color: idle ? "var(--dim)" : fg }}
        >
          {idle ? "—" : value}
        </span>
        {unit && (
          <span className="text-[11px]" style={{ color: idle ? "var(--dim)" : fg }}>
            {unit}
          </span>
        )}
      </div>
      {hint && (
        <div className="mt-2 text-[11px] leading-snug text-[var(--muted)]">{hint}</div>
      )}
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  hint,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-start gap-2.5 ${
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"
      }`}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={`mt-0.5 flex h-[22px] w-[38px] shrink-0 items-center rounded-full transition-colors ${
          checked ? "bg-[var(--text)]" : "bg-[#dcd3c5]"
        }`}
      >
        <span
          className={`block h-[16px] w-[16px] rounded-full bg-white shadow-sm transition-transform ${
            checked ? "translate-x-[19px]" : "translate-x-[3px]"
          }`}
        />
      </button>
      <span>
        <span className="block text-[13px] font-medium text-[var(--text)]">{label}</span>
        {hint && (
          <span className="block text-[11px] leading-snug text-[var(--muted)]">
            {hint}
          </span>
        )}
      </span>
    </label>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
}) {
  const base =
    "rounded-full px-5 py-2.5 text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed";
  const styles =
    variant === "primary"
      ? "bg-[var(--amber)] text-[var(--text)] hover:brightness-[0.97] shadow-[var(--shadow-sm)]"
      : "border border-[var(--line)] bg-[var(--panel)] text-[var(--text)] hover:bg-[var(--panel-2)]";
  return (
    <button onClick={onClick} disabled={disabled} className={`${base} ${styles}`}>
      {children}
    </button>
  );
}

export function Bar({
  value,
  max,
  color = "var(--accent)",
  height = 8,
}: {
  value: number;
  max: number;
  color?: string;
  height?: number;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      className="w-full overflow-hidden rounded-full bg-[var(--panel-2)]"
      style={{ height }}
    >
      <div
        className="h-full rounded-full transition-all duration-200"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

/**
 * Empty state. Rather than a lone glyph in a large void, it ghosts the shape
 * of the thing that's about to appear — so the box reads as "ready" instead of
 * "broken", and the layout doesn't lurch when real data lands.
 */
export function Empty({ children }: { children: ReactNode }) {
  // No ghost bars or placeholder grids here. This app's claim is that every
  // mark on screen is a real measurement, so a skeleton shaped like a chart —
  // with invented heights — would contradict the one thing it's selling.
  // An empty panel says it's empty.
  return (
    <div className="flex min-h-[150px] items-center justify-center px-6 py-4">
      <p className="max-w-[42ch] text-center text-[13px] leading-relaxed text-[var(--muted)]">
        {children}
      </p>
    </div>
  );
}

/**
 * Wraps the last word of a title in the hand-drawn ring, the way the
 * reference does ("Improve your [Skills] Faster").
 */
function ringLastWord(title: string) {
  const words = title.trim().split(" ");
  if (words.length < 2) return <span className="circled">{title}</span>;
  const last = words.pop()!;
  // keep trailing punctuation outside the ring so the oval stays tight
  const m = last.match(/^(.+?)([.?!,]*)$/);
  const word = m ? m[1] : last;
  const punct = m ? m[2] : "";
  return (
    <>
      {words.join(" ")} <span className="circled">{word}</span>
      {punct}
    </>
  );
}

/** Page opener: small kicker, then a headline with its last word circled. */
export function PageHead({
  kicker,
  title,
  children,
}: {
  kicker: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <header className="mb-7">
      <div className="section-label text-[var(--dim)]">{kicker}</div>
      <h2 className="display mt-2 text-[clamp(26px,3.8vw,40px)] leading-[1.16]">
        {ringLastWord(title)}
      </h2>
      {children && (
        <p className="mt-3 max-w-[70ch] text-[14px] leading-relaxed text-[var(--muted)]">
          {children}
        </p>
      )}
    </header>
  );
}

export function Badge({
  children,
  tone = "accent",
}: {
  children: ReactNode;
  tone?: keyof typeof TONES;
  /** kept for API compatibility */
  rotate?: number;
}) {
  const { fg, bg } = TONES[tone];
  return (
    <span
      className="inline-block rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.09em]"
      style={{ background: bg, color: fg }}
    >
      {children}
    </span>
  );
}
