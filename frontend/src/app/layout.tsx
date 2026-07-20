import type { Metadata } from "next";
import { Bricolage_Grotesque, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import Link from "next/link";
import { Wordmark } from "@/components/logo";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--font-bricolage",
});
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
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

/* The ticker carries the project's real measured results — decoration that
   happens to be the headline evidence. */
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
      className={`${bricolage.variable} ${spaceGrotesk.variable} ${jetbrains.variable}`}
    >
      <body className="min-h-screen">
        {/* ---- top marquee ---- */}
        <div className="marquee border-b-2 border-[var(--line)] bg-[var(--line)] py-1.5 text-[var(--bg)]">
          {[0, 1].map((dup) => (
            <div className="marquee__track" key={dup} aria-hidden={dup === 1}>
              {TICKER.map((t, i) => (
                <span
                  key={i}
                  className="mono flex items-center gap-2.5 whitespace-nowrap text-[10px] font-medium tracking-[0.12em]"
                >
                  <span aria-hidden>✦</span>
                  {t}
                </span>
              ))}
            </div>
          ))}
        </div>

        {/* ---- header ---- */}
        <header className="sticky top-0 z-50 border-b-2 border-[var(--line)] bg-[var(--bg)]/92 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
            <Link href="/" aria-label="Chimera — home">
              <Wordmark animate />
            </Link>

            <nav className="ml-auto flex flex-wrap items-center gap-1.5">
              {NAV.map((n) => (
                <Link
                  key={n.href}
                  href={n.href}
                  className="group flex items-center gap-1.5 whitespace-nowrap rounded-full border-2 border-transparent px-3 py-1.5 text-xs font-semibold text-[var(--muted)] transition-all hover:border-[var(--line)] hover:bg-[var(--panel)] hover:text-[var(--text)]"
                >
                  <span className="mono text-[9px] opacity-50">{n.n}</span>
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>

        <main className="mx-auto max-w-[1400px] px-6 py-8">{children}</main>

        <footer className="mt-12 border-t-2 border-[var(--line)] bg-[var(--panel)]">
          <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-6 py-6">
            <div className="display text-[13px] tracking-wide">
              CHIMERA <span className="opacity-40">✦</span>{" "}
              <span className="font-normal opacity-60">
                an LLM inference engine you can watch
              </span>
            </div>
            <a
              href="https://github.com/Komalpreet2809/Chimera"
              target="_blank"
              rel="noreferrer"
              className="sticker px-4 py-1.5 text-[11px] font-bold tracking-wide transition-transform hover:-translate-y-0.5"
            >
              SOURCE ↗
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
