"use client";

import { Icon } from "@/components/ui/icon";
import type { JobDetail } from "@/lib/types";

interface ReferencePaneProps {
  job: JobDetail | null;
  loading: boolean;
  resumePath?: string;
}

export function ReferencePane({ job, loading, resumePath }: ReferencePaneProps) {
  if (loading) {
    return (
      <div className="p-8 space-y-6">
        <div className="h-8 bg-surface-alt animate-pulse w-3/4" />
        <div className="h-4 bg-surface-alt animate-pulse w-1/2" />
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-3 bg-surface-alt animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="p-8 text-center">
        <Icon name="description" size={48} className="text-border mb-4" />
        <p className="text-muted text-[11px] uppercase tracking-widest">
          Select a job from the list above to view details
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto custom-scroll p-8 space-y-12">
      {/* Job Listing Block */}
      <div>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-2 h-2 bg-navy" />
          <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-navy">
            Job Listing
          </h3>
        </div>
        <div className="bg-card p-8 border border-navy/10">
          <h4 className="text-navy font-heading font-bold text-2xl mb-2">
            {job.title}
          </h4>
          <p className="text-muted text-[10px] font-bold uppercase tracking-widest mb-6">
            {job.company || "Unknown"} {job.location ? `• ${job.location}` : ""}{" "}
            {job.salary ? `• ${job.salary}` : ""}
          </p>

          {job.fit_score && (
            <div className="mb-6 flex items-center gap-2">
              <span className="text-[9px] font-bold border border-navy px-2 py-0.5 uppercase">
                MATCH SCORE: {job.fit_score}/10
              </span>
            </div>
          )}

          {job.fit_reason && (
            <div className="mb-6 bg-surface p-4 border border-border-light">
              <p className="text-[10px] text-muted leading-relaxed">
                <span className="font-bold text-navy">AI ASSESSMENT:</span>{" "}
                {job.fit_reason}
              </p>
            </div>
          )}

          {job.description && (
            <div className="text-xs text-navy/80 leading-loose max-h-[300px] overflow-y-auto custom-scroll">
              {job.description.slice(0, 2000)}
              {job.description.length > 2000 && (
                <span className="text-muted"> ... (truncated)</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Base Resume Block */}
      <div>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-2 h-2 bg-navy/40" />
          <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-navy/60">
            Base Resume
          </h3>
        </div>
        <div className="bg-card p-8 border border-navy/5 opacity-60">
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest text-navy">
                {resumePath ? resumePath.split("/").pop() : "No resume uploaded"}
              </p>
            </div>
            <Icon name="lock" size={14} className="text-muted" />
          </div>
          <p className="text-[9px] text-muted uppercase tracking-widest">
            Upload or update your base resume in Settings
          </p>
        </div>
      </div>
    </div>
  );
}
