import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chimera — LLM inference, made visible",
  description:
    "An interactive LLM inference engine: watch KV cache, continuous batching, PagedAttention and speculative decoding actually happen.",
};

const NAV = [
  { href: "/", label: "Inference Lab" },
  { href: "/scheduler", label: "Scheduler" },
  { href: "/memory", label: "Paged Memory" },
  { href: "/attention", label: "Attention" },
  { href: "/benchmarks", label: "Benchmarks" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--bg)]/85 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] items-center gap-6 px-6 py-3">
            <Link href="/" className="flex shrink-0 items-center gap-2.5">
              <div className="grid h-7 w-7 place-items-center rounded-md bg-[var(--accent)] text-sm font-bold text-[#06121f]">
                C
              </div>
              <div className="leading-tight">
                <div className="text-[13px] font-semibold">Chimera</div>
                <div className="text-[10px] text-[var(--dim)]">
                  LLM inference, made visible
                </div>
              </div>
            </Link>
            <nav className="flex items-center gap-1 overflow-x-auto">
              {NAV.map((n) => (
                <Link
                  key={n.href}
                  href={n.href}
                  className="whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-medium text-[var(--muted)] transition-colors hover:bg-[var(--panel)] hover:text-[var(--text)]"
                >
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
