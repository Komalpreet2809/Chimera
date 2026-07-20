/**
 * Chimera mark — the causal-attention staircase drawn as paged KV blocks.
 *
 * Row i attends to columns 0..i: the lower-triangular mask that makes
 * left-to-right generation possible, and the same grid PagedAttention hands
 * out in fixed-size blocks. Two of the engine's core ideas in one shape.
 */

/** Cells of the lower triangle, as [col, row] pairs. */
const CELLS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 1],
  [0, 2],
  [1, 2],
  [2, 2],
];

export function Logo({
  size = 36,
  animate = false,
  className = "",
}: {
  size?: number;
  /** Fill the staircase in sequence, the way tokens actually arrive. */
  animate?: boolean;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 32 32"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label="Chimera"
    >
      <rect
        x="1"
        y="1"
        width="30"
        height="30"
        rx="8"
        fill="var(--brand, #e8a33d)"
        stroke="var(--line, #22271c)"
        strokeWidth="2"
      />
      <g fill="var(--line, #22271c)">
        {CELLS.map(([col, row], i) => (
          <rect
            key={`${col}-${row}`}
            x={5 + col * 8}
            y={5 + row * 8}
            width="6"
            height="6"
            rx="1"
          >
            {animate && (
              <animate
                attributeName="opacity"
                values="0.15; 1; 1; 0.15"
                keyTimes="0; 0.15; 0.85; 1"
                dur="2.4s"
                begin={`${i * 0.18}s`}
                repeatCount="indefinite"
              />
            )}
          </rect>
        ))}
      </g>
    </svg>
  );
}

/** Full lockup: mark + wordmark, for the header. */
export function Wordmark({ animate = false }: { animate?: boolean }) {
  return (
    <span className="flex shrink-0 items-center gap-3">
      <span
        className="grid place-items-center rounded-[10px]"
        style={{ boxShadow: "var(--shadow-sm)", lineHeight: 0 }}
      >
        <Logo size={36} animate={animate} />
      </span>
      <span className="leading-none">
        <span className="display block text-[19px] tracking-tight">CHIMERA</span>
        <span className="mt-1 block text-[10px] uppercase tracking-[0.1em] text-[var(--muted)]">
          LLM inference, made visible
        </span>
      </span>
    </span>
  );
}
