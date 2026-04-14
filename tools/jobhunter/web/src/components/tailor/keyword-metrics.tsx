"use client";

import type { JobDetail } from "@/lib/types";

interface KeywordMetricsProps {
  job: JobDetail | null;
}

export function KeywordMetrics({ job }: KeywordMetricsProps) {
  if (!job) return null;

  // Extract keywords from fit_reason if available
  const keywords: { term: string; match: number }[] = [];
  if (job.fit_reason) {
    // Generate pseudo-metrics from the fit score and reason text
    const words = job.fit_reason
      .split(/[\s,;.]+/)
      .filter((w) => w.length > 4)
      .map((w) => w.toLowerCase())
      .filter((w, i, arr) => arr.indexOf(w) === i)
      .slice(0, 4);

    const baseScore = ((job.fit_score ?? 5) / 10) * 100;
    words.forEach((word, i) => {
      keywords.push({
        term: word.charAt(0).toUpperCase() + word.slice(1),
        match: Math.max(20, Math.min(99, baseScore - i * 12 + Math.random() * 10)),
      });
    });
  }

  if (keywords.length === 0) return null;

  return (
    <div className="bg-card border border-navy p-6 hidden xl:block">
      <h4 className="text-[10px] font-bold text-navy uppercase tracking-[0.2em] mb-6">
        Metrics
      </h4>
      <div className="space-y-6">
        {keywords.map((kw) => (
          <div key={kw.term} className="space-y-2">
            <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest">
              <span>{kw.term}</span>
              <span>{Math.round(kw.match)}%</span>
            </div>
            <div className="h-[1px] w-full bg-navy/10">
              <div
                className="h-full bg-navy"
                style={{ width: `${kw.match}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <button className="w-full mt-8 py-3 border border-navy text-[9px] font-bold uppercase tracking-widest hover:bg-navy hover:text-white transition-sharp">
        Full Analysis
      </button>
    </div>
  );
}
