"use client";

import { useRef, useEffect } from "react";

export interface LogEntry {
  timestamp: string;
  level: "info" | "warn" | "error" | "success";
  message: string;
}

interface TerminalFeedProps {
  entries: LogEntry[];
  title?: string;
  height?: string;
  showCursor?: boolean;
  variant?: "light" | "dark";
}

export function TerminalFeed({
  entries,
  title = "SYSTEM LOG",
  height = "h-64",
  showCursor = true,
  variant = "light",
}: TerminalFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  if (variant === "dark") {
    return (
      <div className="bg-navy border border-navy-light">
        <div className="px-5 py-3 border-b border-navy-light flex justify-between items-center">
          <span className="font-mono text-[9px] font-semibold tracking-widest text-primary-light uppercase">
            {title}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-accent animate-pulse" />
            <span className="font-mono text-[9px] text-muted">LIVE</span>
          </span>
        </div>
        <div
          ref={scrollRef}
          className={`p-5 font-mono text-[10px] leading-relaxed ${height} overflow-y-auto terminal-scroll`}
        >
          <div className="space-y-1.5">
            {entries.map((entry, i) => (
              <p key={i} className={
                entry.level === "error" ? "text-danger" :
                entry.level === "success" ? "text-accent" :
                entry.level === "warn" ? "text-warning" :
                "text-border"
              }>
                <span className="text-muted">[{entry.timestamp}]</span>{" "}
                {entry.message}
              </p>
            ))}
            {showCursor && (
              <p className="text-primary-light animate-pulse">_</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-card border border-border-light">
      <div className="flex items-center justify-between px-5 py-3 border-b border-border-light">
        <span className="font-mono text-[9px] font-semibold text-muted tracking-widest uppercase">
          {title}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 bg-accent animate-pulse" />
          <span className="font-mono text-[9px] text-accent font-semibold">ACTIVE</span>
        </span>
      </div>
      <div
        ref={scrollRef}
        className={`p-5 font-mono text-[10px] leading-relaxed ${height} overflow-y-auto terminal-scroll`}
      >
        <div className="space-y-1.5">
          {entries.map((entry, i) => (
            <p key={i} className={
              entry.level === "error" ? "text-danger font-semibold" :
              entry.level === "success" ? "text-accent-dark font-semibold" :
              entry.level === "warn" ? "text-warning" :
              "text-text-light"
            }>
              <span className="text-muted">[{entry.timestamp}]</span>{" "}
              {entry.message}
            </p>
          ))}
          {showCursor && (
            <span className="inline-block w-2 h-3.5 bg-primary animate-pulse" />
          )}
        </div>
      </div>
    </div>
  );
}
