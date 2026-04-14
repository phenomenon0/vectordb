"use client";

import Link from "next/link";
import { Icon } from "@/components/ui/icon";
import type { JobRow } from "@/lib/types";

interface KanbanCardProps {
  job: JobRow;
  priority?: boolean;
}

export function KanbanCard({ job, priority = false }: KanbanCardProps) {
  const dateStr = new Date(job.scraped_at).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  }).toUpperCase();

  return (
    <Link
      href={`/tailor?id=${job.id}`}
      className={`bg-card p-6 hover:bg-surface transition-sharp group cursor-pointer block ${
        priority ? "border-2 border-navy" : "border border-navy"
      }`}
    >
      <div className="flex justify-between items-start mb-4">
        <span className="text-[9px] font-mono font-bold">JH-{job.id}</span>
        <span className="text-[9px] font-bold text-muted">{dateStr}</span>
      </div>
      <h4 className="font-heading font-bold text-navy text-lg leading-tight">
        {job.title}
      </h4>
      <p className="text-[10px] text-muted uppercase tracking-widest mt-2">
        {job.company || "Unknown"}
      </p>
      <div className="mt-6 pt-4 border-t border-border-light flex items-center justify-between">
        {job.salary ? (
          <div className="text-[11px] font-bold">{job.salary}</div>
        ) : (
          <span className="text-[9px] font-bold text-muted uppercase">
            {job.fit_score ? `SCORE: ${job.fit_score}/10` : "PENDING"}
          </span>
        )}
        {priority && (
          <span className="text-[9px] font-bold px-2 py-0.5 bg-navy text-white uppercase">
            Priority
          </span>
        )}
      </div>
    </Link>
  );
}
