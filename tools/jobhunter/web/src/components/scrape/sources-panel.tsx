"use client";

import { Icon } from "@/components/ui/icon";

const SOURCES = [
  { key: "linkedin", label: "LinkedIn", letter: "L" },
  { key: "indeed", label: "Indeed", letter: "I" },
  { key: "glassdoor", label: "Glassdoor", letter: "G" },
];

interface SourcesPanelProps {
  activeSites: string[];
  onToggle: (site: string) => void;
  scrapeProgress: number | null; // null = not scraping
}

export function SourcesPanel({ activeSites, onToggle, scrapeProgress }: SourcesPanelProps) {
  return (
    <div className="bg-card p-6 border border-navy">
      <div className="flex justify-between items-center mb-8">
        <h3 className="font-heading font-bold text-navy flex items-center gap-2 uppercase tracking-widest text-sm">
          <Icon name="sensors" size={16} />
          Active Sources
        </h3>
        <span className="text-[9px] text-navy font-bold uppercase tracking-widest">
          {scrapeProgress !== null ? "Status: Active" : "Status: Ready"}
        </span>
      </div>

      <div className="space-y-8">
        {SOURCES.map((source) => {
          const isActive = activeSites.includes(source.key);
          return (
            <div
              key={source.key}
              className={`space-y-4 ${!isActive ? "opacity-30" : ""}`}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 border border-navy flex items-center justify-center font-bold text-navy text-[10px]">
                    {source.letter}
                  </div>
                  <span className="text-[10px] font-bold uppercase tracking-widest">
                    {source.label}
                  </span>
                </div>
                <button
                  onClick={() => onToggle(source.key)}
                  className="w-8 h-4 border border-navy relative cursor-pointer"
                >
                  <div
                    className={`absolute top-0 bottom-0 w-4 transition-sharp ${
                      isActive ? "right-0 bg-navy" : "left-0 bg-surface-alt"
                    }`}
                  />
                </button>
              </div>
              <div className="w-full bg-surface-alt h-[2px] relative">
                {isActive && scrapeProgress !== null && (
                  <div
                    className="bg-navy h-full transition-sharp"
                    style={{ width: `${scrapeProgress}%` }}
                  />
                )}
              </div>
              <div className="flex justify-between text-[9px] text-muted font-mono uppercase tracking-tighter">
                <span>{isActive ? (scrapeProgress !== null ? "Syncing..." : "Ready") : "Idle"}</span>
                <span>{isActive ? "Active" : "Standby"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
