"use client";

import { ReactNode } from "react";

/** A boxed article / sidebar, with a ruled head. */
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
        <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] bg-[var(--panel-2)] px-4 py-2.5">
          <div>
            {title && <h3 className="section-label text-[var(--text)]">{title}</h3>}
            {subtitle && (
              <p className="serif mt-1 text-[13px] italic leading-snug text-[var(--muted)]">
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

/** A figure block: label above a rule, the number set large below it. */
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
    <div className="border border-[var(--line)] px-3 py-2.5" style={{ background: bg }}>
      <div className="section-label text-[var(--muted)]">{label}</div>
      <div className="mt-1.5 border-t border-[var(--line-soft)] pt-1.5">
        <div className="mono flex items-baseline gap-1">
          <span
            className="text-[24px] font-bold leading-none tabular-nums"
            style={{ color: fg }}
          >
            {value}
          </span>
          {unit && <span className="text-[11px] text-[var(--muted)]">{unit}</span>}
        </div>
      </div>
      {hint && (
        <div className="serif mt-1 text-[12px] italic leading-snug text-[var(--muted)]">
          {hint}
        </div>
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
        className={`mt-0.5 grid h-[18px] w-[18px] shrink-0 place-items-center border border-[var(--line)] transition-colors ${
          checked ? "bg-[var(--line)]" : "bg-[var(--panel)]"
        }`}
      >
        {checked && <span className="text-[11px] leading-none text-[var(--bg)]">✓</span>}
      </button>
      <span>
        <span className="serif block text-[14px] font-semibold text-[var(--text)]">
          {label}
        </span>
        {hint && (
          <span className="serif block text-[12px] italic leading-snug text-[var(--muted)]">
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
    "border border-[var(--line)] px-4 py-2 text-[10px] font-bold uppercase tracking-[0.18em] transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const styles =
    variant === "primary"
      ? "bg-[var(--line)] text-[var(--bg)] hover:bg-[var(--muted)]"
      : "bg-[var(--panel)] text-[var(--text)] hover:bg-[var(--panel-2)]";
  return (
    <button onClick={onClick} disabled={disabled} className={`mono ${base} ${styles}`}>
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
      className="w-full overflow-hidden border border-[var(--line)] bg-[var(--panel-2)]"
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
    <div className="serif flex min-h-[120px] flex-col items-center justify-center gap-2 px-6 text-center text-[14px] italic text-[var(--muted)]">
      <span className="text-lg not-italic opacity-40" aria-hidden>
        ❖
      </span>
      {children}
    </div>
  );
}

/**
 * A magazine feature opener: kicker rule, big serif headline, and a dek set
 * with a drop cap.
 */
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
      <div className="flex items-center gap-3">
        <span className="section-label whitespace-nowrap text-[var(--muted)]">
          {kicker}
        </span>
        <span className="h-px flex-1 bg-[var(--line-soft)]" />
      </div>

      <h2 className="display mt-2.5 text-[clamp(30px,5.2vw,54px)] leading-[0.95]">
        {title}
      </h2>

      {children && (
        <div className="rule-thick mt-3.5 pt-3.5">
          <p className="serif dropcap max-w-[68ch] text-[16px] leading-[1.55] text-[var(--muted)]">
            {children}
          </p>
        </div>
      )}
    </header>
  );
}

/** A small boxed tag, set in mono small caps. */
export function Badge({
  children,
  tone = "accent",
}: {
  children: ReactNode;
  tone?: keyof typeof TONES;
  /** kept for API compatibility; print doesn't rotate its rules */
  rotate?: number;
}) {
  const { fg, bg } = TONES[tone];
  return (
    <span
      className="mono inline-block border border-[var(--line)] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.16em]"
      style={{ background: bg, color: fg }}
    >
      {children}
    </span>
  );
}
