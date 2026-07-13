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
        <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] px-4 py-3">
          <div>
            {title && (
              <h3 className="text-[13px] font-semibold tracking-wide text-[var(--text)]">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="mt-0.5 text-[11px] leading-snug text-[var(--dim)]">
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
  tone?: "default" | "good" | "bad" | "warn" | "accent" | "violet";
  hint?: string;
}) {
  const color = {
    default: "var(--text)",
    good: "var(--good)",
    bad: "var(--bad)",
    warn: "var(--warn)",
    accent: "var(--accent)",
    violet: "var(--violet)",
  }[tone];
  return (
    <div className="rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--dim)]">
        {label}
      </div>
      <div className="mono mt-1 flex items-baseline gap-1">
        <span className="text-xl font-semibold tabular-nums" style={{ color }}>
          {value}
        </span>
        {unit && <span className="text-[11px] text-[var(--muted)]">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-[10px] text-[var(--dim)]">{hint}</div>}
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
      className={`flex cursor-pointer items-start gap-2.5 ${
        disabled ? "cursor-not-allowed opacity-50" : ""
      }`}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={`mt-0.5 h-[18px] w-[32px] shrink-0 rounded-full border transition-colors ${
          checked
            ? "border-[var(--accent)] bg-[var(--accent)]/30"
            : "border-[var(--line)] bg-[var(--panel-2)]"
        }`}
      >
        <span
          className={`block h-3.5 w-3.5 rounded-full transition-transform ${
            checked
              ? "translate-x-[15px] bg-[var(--accent)]"
              : "translate-x-[2px] bg-[var(--dim)]"
          }`}
        />
      </button>
      <span>
        <span className="block text-xs font-medium text-[var(--text)]">{label}</span>
        {hint && (
          <span className="block text-[10px] leading-snug text-[var(--dim)]">{hint}</span>
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
    "rounded-md px-3.5 py-2 text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const styles =
    variant === "primary"
      ? "bg-[var(--accent)] text-[#06121f] hover:bg-[#6fb6ff]"
      : "border border-[var(--line)] text-[var(--muted)] hover:border-[var(--dim)] hover:text-[var(--text)]";
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
  height = 6,
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

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-[120px] items-center justify-center text-center text-xs text-[var(--dim)]">
      {children}
    </div>
  );
}
