"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Inference Lab" },
  { href: "/scheduler", label: "Scheduler" },
  { href: "/memory", label: "Paged Memory" },
  { href: "/attention", label: "Attention" },
  { href: "/benchmarks", label: "Benchmarks" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-wrap items-center gap-1">
      {NAV.map((n) => {
        // "/" would prefix-match everything, so it has to match exactly.
        const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
        return (
          <Link
            key={n.href}
            href={n.href}
            aria-current={active ? "page" : undefined}
            className={`whitespace-nowrap rounded-full px-3.5 py-1.5 text-[13px] transition-colors ${
              active
                ? // A light pill on the amber band reads as "you are here" without
                  // competing with the dark Source button beside it.
                  "bg-[var(--bg)] font-semibold text-[var(--text)] shadow-[var(--shadow-sm)]"
                : "font-medium text-[var(--text)]/70 hover:bg-[#1a1a18]/10 hover:text-[var(--text)]"
            }`}
          >
            {n.label}
          </Link>
        );
      })}
    </nav>
  );
}
