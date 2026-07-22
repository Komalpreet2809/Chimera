"use client";

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { TokenMsg } from "@/lib/api";

export type RuntimeToken = TokenMsg & { elapsed_ms: number };

type RuntimeSession = {
  prompt: string;
  tokens: RuntimeToken[];
  selected: number;
  setTrace: (prompt: string, tokens: RuntimeToken[]) => void;
  appendToken: (prompt: string, token: RuntimeToken) => void;
  select: (index: number) => void;
  clear: () => void;
};

const RuntimeContext = createContext<RuntimeSession | null>(null);

export function RuntimeProvider({ children }: { children: ReactNode }) {
  const [prompt, setPrompt] = useState("");
  const [tokens, setTokens] = useState<RuntimeToken[]>([]);
  const [selected, setSelected] = useState(0);
  // While a trace streams in, the selection follows the newest token — but the
  // moment the user picks one themselves, stop yanking it away from them.
  const following = useRef(true);

  const setTrace = useCallback((nextPrompt: string, nextTokens: RuntimeToken[]) => {
    setPrompt(nextPrompt);
    setTokens(nextTokens);
    setSelected(Math.max(0, nextTokens.length - 1));
    following.current = true;
  }, []);

  const appendToken = useCallback((nextPrompt: string, token: RuntimeToken) => {
    setPrompt(nextPrompt);
    setTokens((current) => {
      // The engine frees a request's cache the moment it finishes, so the
      // FINAL token reports no blocks at all. The allocation didn't vanish
      // mid-trace — the request ended — so carry the last known blocks
      // forward rather than letting the last row render as "—".
      const blocks = token.blocks?.length ? token.blocks : current.at(-1)?.blocks ?? null;
      return [...current, { ...token, blocks }];
    });
    if (following.current) setSelected(token.index - 1);
  }, []);

  const select = useCallback((index: number) => {
    following.current = false;
    setSelected(Math.max(0, index));
  }, []);
  const clear = useCallback(() => {
    setTokens([]);
    setSelected(0);
    following.current = true;
  }, []);

  // Step the trace with the arrow keys — the thing you reach for in any
  // debugger. Ignored while typing, so the prompt box still works normally.
  useEffect(() => {
    if (tokens.length === 0) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      const typing =
        el &&
        (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
      if (typing) return;
      event.preventDefault();
      setSelected((current) =>
        event.key === "ArrowLeft"
          ? Math.max(0, current - 1)
          : Math.min(tokens.length - 1, current + 1)
      );
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [tokens.length]);

  const value = useMemo(
    () => ({ prompt, tokens, selected, setTrace, appendToken, select, clear }),
    [prompt, tokens, selected, setTrace, appendToken, select, clear]
  );

  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

export function useRuntime() {
  const value = useContext(RuntimeContext);
  if (!value) throw new Error("useRuntime must be used inside RuntimeProvider");
  return value;
}
