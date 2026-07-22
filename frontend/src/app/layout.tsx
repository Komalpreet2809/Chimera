import type { Metadata } from "next";
import { Instrument_Serif, JetBrains_Mono, Poppins } from "next/font/google";
import Link from "next/link";
import { Logo } from "@/components/logo";
import { Nav } from "@/components/nav";
import { RuntimeProvider } from "@/components/runtime";
import { RuntimeDock } from "@/components/runtime-debugger";
import "./globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-poppins",
});
const instrument = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: "italic",
  variable: "--font-instrument",
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

/* The headline numbers, all measured — see the Benchmarks page. */
const STATS = [
  { value: "5e-5", label: "logit accuracy" },
  { value: "46×", label: "faster decode" },
  { value: "3.7×", label: "throughput" },
  { value: "4.6×", label: "more users" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${poppins.variable} ${instrument.variable} ${jetbrains.variable}`}
    >
      <body className="min-h-screen">
        <RuntimeProvider>
        {/* ---- amber hero band ---- */}
        <div className="relative overflow-hidden bg-[var(--amber)]">

          <header className="relative mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-3 px-6 py-4">
            <Link href="/" className="flex shrink-0 items-center gap-2.5">
              <Logo size={30} animate />
              <span className="display text-[18px] font-semibold tracking-tight">
                Chimera
              </span>
            </Link>

            <Nav />

            <a
              href="https://github.com/Komalpreet2809/Chimera"
              target="_blank"
              rel="noreferrer"
              className="ml-auto rounded-full bg-[var(--text)] px-4 py-2 text-[13px] font-medium text-white transition-opacity hover:opacity-85"
            >
              Source ↗
            </a>
          </header>

          {/* headline */}
          <div className="relative mx-auto max-w-[1400px] px-6 pb-9 pt-4">
            <h1 className="display max-w-[19ch] text-[clamp(30px,4.6vw,52px)] leading-[1.12]">
              See what an LLM actually{" "}
              <span className="circled">does</span> when it
              generates a token.
            </h1>
            <p className="mt-3.5 max-w-[62ch] text-[14px] leading-relaxed text-[var(--text)]/70">
              A GPT-2 built from scratch — attention, KV cache, continuous batching,
              PagedAttention and speculative decoding, all hand-written — streaming
              its own internals live as it runs.
            </p>

            {/* stat strip */}
            <div className="mt-7 flex flex-wrap items-center gap-y-4">
              {STATS.map((s, i) => (
                <div
                  key={s.label}
                  className={`pr-7 ${i > 0 ? "border-l border-[#1a1a18]/20 pl-7" : ""}`}
                >
                  <div className="mono text-[22px] font-bold leading-none tabular-nums">
                    {s.value}
                  </div>
                  <div className="mt-1 text-[11px] text-[var(--text)]/60">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mx-auto max-w-[1400px] px-6"><RuntimeDock /></div>
        <main className="mx-auto max-w-[1400px] px-6 py-9">{children}</main>

        <footer className="mt-14 bg-[var(--dark)] text-[#f6f1e8]">
          <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-5 px-6 py-9">
            <div>
              <div className="display flex items-center gap-2.5 text-[17px] font-semibold">
                <Logo size={26} />
                Chimera
              </div>
              <p className="mt-2 max-w-[46ch] text-[13px] leading-relaxed text-white/55">
                An LLM inference engine built from first principles, and the
                instruments to watch it run.
              </p>
            </div>
            <a
              href="https://github.com/Komalpreet2809/Chimera"
              target="_blank"
              rel="noreferrer"
              className="rounded-full bg-[var(--amber)] px-5 py-2.5 text-[13px] font-medium text-[var(--text)] transition-opacity hover:opacity-90"
            >
              View source ↗
            </a>
          </div>
        </footer>
        </RuntimeProvider>
      </body>
    </html>
  );
}
