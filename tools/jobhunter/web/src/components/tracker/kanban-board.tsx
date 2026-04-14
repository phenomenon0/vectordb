"use client";

import { useMemo, useCallback } from "react";
import { KanbanColumn } from "./kanban-column";
import type { JobRow, TrackingStatus } from "@/lib/types";
import { KANBAN_COLUMNS } from "@/lib/types";

interface KanbanBoardProps {
  jobs: JobRow[];
  onStatusChange: (jobId: number, status: TrackingStatus) => void;
}

export function KanbanBoard({ jobs, onStatusChange }: KanbanBoardProps) {
  const columns = useMemo(() => {
    const grouped: Record<string, JobRow[]> = {};
    for (const col of KANBAN_COLUMNS) {
      grouped[col.key] = [];
    }
    for (const job of jobs) {
      if (grouped[job.status]) {
        grouped[job.status].push(job);
      }
    }
    return grouped;
  }, [jobs]);

  const handleDrop = useCallback(
    (jobId: number, status: TrackingStatus) => {
      onStatusChange(jobId, status);
    },
    [onStatusChange]
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 items-start">
      {KANBAN_COLUMNS.map((col) => (
        <KanbanColumn
          key={col.key}
          title={col.label}
          status={col.key}
          jobs={columns[col.key] || []}
          onDrop={handleDrop}
          faded={col.key === "rejected"}
        />
      ))}
    </div>
  );
}
