"use client";

import { Icon } from "@/components/ui/icon";
import type { JobRow } from "@/lib/types";

interface JobPickerProps {
  jobs: JobRow[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export function JobPicker({ jobs, selectedId, onSelect }: JobPickerProps) {
  if (jobs.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted text-[11px] uppercase tracking-widest">
          No jobs ready for tailoring. Run a scrape and filter cycle first.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scroll">
      {jobs.map((job) => (
        <button
          key={job.id}
          onClick={() => onSelect(job.id)}
          className={`w-full text-left p-4 transition-sharp ${
            selectedId === job.id
              ? "bg-navy text-white"
              : "bg-card hover:bg-surface border border-border-light"
          }`}
        >
          <div className="flex justify-between items-start">
            <div>
              <p className={`font-heading font-bold text-sm ${selectedId === job.id ? "text-white" : "text-navy"}`}>
                {job.title}
              </p>
              <p className={`text-[9px] uppercase tracking-widest mt-1 ${selectedId === job.id ? "text-white/60" : "text-muted"}`}>
                {job.company || "Unknown"}
              </p>
            </div>
            {job.fit_score && (
              <span className={`text-[9px] font-mono font-bold ${selectedId === job.id ? "text-white" : "text-navy"}`}>
                {job.fit_score}/10
              </span>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}
