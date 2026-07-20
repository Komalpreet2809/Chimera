import type { Metadata } from "next";
import { EB_Garamond, JetBrains_Mono, Playfair_Display } from "next/font/google";
import Link from "next/link";
import { Logo } from "@/components/logo";
import "./globals.css";

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
});
const garamond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-garamond",
});
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "Chimera — LLM inference, made visible",
  description:
    "An interactive LLM inference engine: watch KV cache, continuous batching, PagedAttention and speculative decoding actually happen.",
};

const NAV = [
  { href: "/", label: "Inference Lab", n: "01" },
  { href: "/scheduler", label: "Scheduler", n: "02" },
  { href: "/memory", label: "Paged Memory", n: "03" },
  { href: "/attention", label: "Attention", n: "04" },
  { href: "/benchmarks", label: "Benchmarks", n: "05" },
];

/* Set as a wire-service ticker — decoration that happens to be the evidence. */
const TICKER = [
  "GPT-2 BUILT FROM SCRATCH",
  "5e-5 LOGIT ACCURACY",
  "KV CACHE — 46× FASTER DECODE",
  "CONTINUOUS BATCHING — 3.7× THROUGHPUT",
  "TTFT 25.8s → 3.4s",
  "PAGEDATTENTION — 80% → 6% WASTE",
  "4.6× MORE CONCURRENT USERS",
  "SPECULATIVE DECODING",
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${playfair.variable} ${garamond.variable} ${jetbrains.variable}`}
    >
      <body className="min-h-screen">
        {/* ---- wire ticker ---- */}
        <div className="marquee border-b border-[var(--line)] bg-[var(--line)] py-1 text-[var(--bg)]">
          {[0, 1].map((dup) => (
            <div className="marquee__track" key={dup} aria-hidden={dup === 1}>
              {TICKER.map((t, i) => (
                <span
                  key={i}
                  className="mono flex items-center gap-2.5 whitespace-nowrap text-[9px] tracking-[0.2em]"
                >
                  <span aria-hidden>❖</span>
                  {t}
                </span>
              ))}
            </div>
          ))}
        </div>

        {/* ---- masthead ---- */}
        <div className="border-b border-[var(--line-soft)] bg-[var(--panel)]">
          <div className="mx-auto max-w-[1400px] px-6">
            {/* running head */}
            <div className="flex items-center justify-between gap-4 py-1.5">
              <span className="folio">Vol. I · No. 1</span>
              <span className="folio hidden sm:block">
                An Illustrated Guide to Machine Inference
              </span>
              <span className="folio">Est. 2026</span>
            </div>

            <div className="rule" />

            {/* the masthead proper */}
            <Link href="/" aria-label="Chimera — home" className="block">
              <div className="flex items-center justify-center gap-5 py-5">
                <Logo size={44} animate className="hidden shrink-0 sm:block" />
                <h1 className="display text-center text-[clamp(38px,8vw,84px)] leading-[0.86] tracking-[0.01em]">
                  Chimera
                </h1>
                <Logo size={44} animate className="hidden shrink-0 sm:block" />
              </div>
              <p className="serif -mt-1 pb-4 text-center text-[13px] italic tracking-wide text-[var(--muted)]">
                LLM inference, made visible — an engine built from first
                principles, and the instruments to watch it run
              </p>
            </Link>
          </div>
        </div>

        {/* ---- contents bar ---- */}
        <nav className="rule-double sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--bg)]/95 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] flex-wrap items-stretch justify-center divide-x divide-[var(--line-soft)] px-6">
            {NAV.map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className="group flex items-baseline gap-2 px-4 py-2.5 transition-colors hover:bg-[var(--panel-2)]"
              >
                <span className="mono text-[9px] tracking-[0.15em] text-[var(--dim)]">
                  {n.n}
                </span>
                <span className="serif text-[15px] text-[var(--text)] group-hover:italic">
                  {n.label}
                </span>
              </Link>
            ))}
          </div>
        </nav>

        <main className="mx-auto max-w-[1400px] px-6 py-8">{children}</main>

        <footer className="mt-14 border-t border-[var(--line)] bg-[var(--panel)]">
          <div className="mx-auto max-w-[1400px] px-6 py-7">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="serif text-[14px] italic text-[var(--muted)]">
                Printed from a GPT-2 written by hand, in{" "}
                <span className="not-italic">PyTorch</span>.
              </div>
              <a
                href="https://github.com/Komalpreet2809/Chimera"
                target="_blank"
                rel="noreferrer"
                className="sticker px-3.5 py-1.5 text-[10px] font-bold uppercase tracking-[0.18em] transition-colors hover:bg-[var(--panel-2)]"
              >
                Source ↗
              </a>
            </div>
            <div className="rule mt-5 pt-3">
              <p className="folio text-center">
                Chimera · An Illustrated Guide to Machine Inference · Vol. I
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
