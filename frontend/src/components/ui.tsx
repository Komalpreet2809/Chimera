"use client";

import { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`panel ${className}`}>
      {(title || right) && (
        <div className="flex items-start justify-between gap-4 border-b-2 border-[var(--line)] px-4 py-3">
          <div>
            {title && <h3 className="section-label text-[var(--text)]">{title}</h3>}
            {subtitle && (
              <p className="mt-1 text-[11px] leading-snug text-[var(--muted)]">
                {subtitle}
              </p>
            )}
          </div>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
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
  return (
    <div
      className="rounded-[10px] border-2 border-[var(--line)] px-3 py-2.5"
      style={{ background: bg, boxShadow: "var(--shadow-sm)" }}
    >
      <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-[var(--muted)]">
        {label}
      </div>
      <div className="mono mt-1.5 flex items-baseline gap-1">
        <span className="text-[22px] font-bold tabular-nums leading-none" style={{ color: fg }}>
          {value}
        </span>
        {unit && (
          <span className="text-[11px] font-medium text-[var(--muted)]">{unit}</span>
        )}
      </div>
      {hint && <div className="mt-1 text-[10px] text-[var(--muted)]">{hint}</div>}
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
        className={`mt-0.5 flex h-[20px] w-[36px] shrink-0 items-center rounded-full border-2 border-[var(--line)] transition-colors ${
          checked ? "bg-[var(--good)]" : "bg-[var(--panel-2)]"
        }`}
      >
        <span
          className={`block h-3 w-3 rounded-full border-2 border-[var(--line)] bg-[var(--bg)] transition-transform ${
            checked ? "translate-x-[17px]" : "translate-x-[2px]"
          }`}
        />
      </button>
      <span>
        <span className="block text-xs font-bold text-[var(--text)]">{label}</span>
        {hint && (
          <span className="block text-[10px] leading-snug text-[var(--muted)]">
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
    "rounded-full border-2 border-[var(--line)] px-4 py-2 text-xs font-bold tracking-wide transition-all active:translate-x-[2px] active:translate-y-[2px] active:shadow-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0";
  const styles =
    variant === "primary"
      ? "bg-[var(--warn)] text-[var(--line)] hover:-translate-y-0.5"
      : "bg-[var(--panel)] text-[var(--text)] hover:-translate-y-0.5";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${styles}`}
      style={{ boxShadow: disabled ? "none" : "var(--shadow-sm)" }}
    >
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
      className="w-full overflow-hidden rounded-full border-2 border-[var(--line)] bg-[var(--panel-2)]"
      style={{ height }}
    >
      <div
        className="h-full transition-all duration-200"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-[120px] flex-col items-center justify-center gap-2 text-center text-xs text-[var(--muted)]">
      <span className="text-lg opacity-40" aria-hidden>
        ✦
      </span>
      {children}
    </div>
  );
}

/** Editorial page header: big display title + kicker. */
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
    <div className="mb-6">
      <div className="section-label text-[var(--muted)]">{kicker}</div>
      <h1 className="display mt-1.5 text-[clamp(28px,4.4vw,46px)] leading-[0.95]">
        {title}
      </h1>
      {children && (
        <p className="mt-2.5 max-w-[70ch] text-[13px] leading-relaxed text-[var(--muted)]">
          {children}
        </p>
      )}
    </div>
  );
}

/** Small rotated sticker badge, for callouts. */
export function Badge({
  children,
  tone = "accent",
  rotate = -2,
}: {
  children: ReactNode;
  tone?: keyof typeof TONES;
  rotate?: number;
}) {
  const { fg, bg } = TONES[tone];
  return (
    <span
      className="sticker inline-block px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.1em]"
      style={{ background: bg, color: fg, transform: `rotate(${rotate}deg)` }}
    >
      {children}
    </span>
  );
}
