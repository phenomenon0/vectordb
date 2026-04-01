"use client";

import { Icon } from "@/components/ui/icon";
import { KanbanCard } from "./kanban-card";
import type { JobRow, TrackingStatus } from "@/lib/types";

interface KanbanColumnProps {
  title: string;
  status: TrackingStatus;
  jobs: JobRow[];
  onDrop: (jobId: number, status: TrackingStatus) => void;
  faded?: boolean;
}

export function KanbanColumn({ title, status, jobs, onDrop, faded = false }: KanbanColumnProps) {
  return (
    <div
      className={`flex flex-col gap-6 ${faded ? "grayscale opacity-40 hover:opacity-100 transition-all duration-300" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        e.currentTarget.classList.add("bg-surface-alt/50");
      }}
      onDragLeave={(e) => {
        e.currentTarget.classList.remove("bg-surface-alt/50");
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.currentTarget.classList.remove("bg-surface-alt/50");
        const jobId = parseInt(e.dataTransfer.getData("text/plain"), 10);
        if (!isNaN(jobId)) onDrop(jobId, status);
      }}
    >
      <div className="flex items-center justify-between border-b border-navy pb-2">
        <div className="flex items-center gap-2">
          <h3 className="font-heading font-bold text-navy tracking-wider uppercase text-[11px]">
            {title}
          </h3>
          <span className="text-[9px] border border-navy px-1.5 font-bold">
            {jobs.length}
          </span>
        </div>
        <Icon name="more_vert" size={14} className="text-muted" />
      </div>
      <div className="space-y-6 min-h-[100px]">
        {jobs.map((job) => (
          <div
            key={job.id}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData("text/plain", String(job.id));
              e.dataTransfer.effectAllowed = "move";
            }}
          >
            <KanbanCard
              job={job}
              priority={status === "interviewing" || status === "offer"}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
