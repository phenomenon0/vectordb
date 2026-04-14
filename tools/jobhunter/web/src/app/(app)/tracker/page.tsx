"use client";

import { useState, useCallback, useMemo } from "react";
import { Topbar } from "@/components/layout/topbar";
import { useJobs, useStats, useTerminalLog } from "@/lib/hooks";
import { KanbanBoard } from "@/components/tracker/kanban-board";
import { TerminalFeed } from "@/components/ui/terminal-feed";
import { Icon } from "@/components/ui/icon";
import { Button } from "@/components/ui/button";
import type { TrackingStatus, JobRow } from "@/lib/types";
import { api } from "@/lib/api";
import Link from "next/link";

export default function TrackerPage() {
  const { jobs, refresh } = useJobs();
  const { stats } = useStats();
  const { entries } = useTerminalLog(stats);
  const [view, setView] = useState<"kanban" | "table">("kanban");

  // Include tailored jobs as "ready to apply" + all tracking statuses
  const trackerJobs = useMemo(() => {
    return jobs.filter(
      (j) =>
        j.status === "applied" ||
        j.status === "interviewing" ||
        j.status === "offer" ||
        j.status === "rejected" ||
        j.status === "withdrawn"
    );
  }, [jobs]);

  // Ready to apply (tailored but not yet in tracker)
  const readyJobs = useMemo(() => {
    return jobs.filter((j) => j.status === "tailored");
  }, [jobs]);

  const handleStatusChange = useCallback(
    async (jobId: number, status: TrackingStatus) => {
      try {
        const res = await fetch(api(`/api/jobs/${jobId}/status`), {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status }),
        });
        if (!res.ok) throw new Error("Failed to update");
        await refresh();
      } catch (e) {
        console.error("Status update failed:", e);
      }
    },
    [refresh]
  );

  // Tracker metrics
  const appliedCount = trackerJobs.filter((j) => j.status === "applied").length;
  const interviewingCount = trackerJobs.filter((j) => j.status === "interviewing").length;
  const offerCount = trackerJobs.filter((j) => j.status === "offer").length;
  const responseRate = stats.total > 0 ? Math.round((appliedCount / stats.total) * 100) : 0;
  const conversionRate = appliedCount > 0 ? Math.round((interviewingCount / appliedCount) * 100) : 0;

  return (
    <>
      <Topbar title="Application Tracker" />

      <div className="flex-1 overflow-y-auto custom-scroll p-12 space-y-12">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-8 border-b border-navy">
          <div>
            <h2 className="font-heading text-5xl font-bold text-navy tracking-tight">
              Application Tracker
            </h2>
            <p className="text-muted font-sans text-xs uppercase tracking-[0.2em] mt-4">
              Monitoring {trackerJobs.length} active recruitment pipelines
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* View Toggle */}
            <div className="bg-card border border-navy flex">
              <button
                onClick={() => setView("kanban")}
                className={`px-6 py-2 text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 ${
                  view === "kanban"
                    ? "bg-navy text-white"
                    : "text-navy hover:bg-surface"
                }`}
              >
                Kanban
              </button>
              <button
                onClick={() => setView("table")}
                className={`px-6 py-2 text-[10px] uppercase tracking-widest font-bold flex items-center gap-2 border-l border-navy ${
                  view === "table"
                    ? "bg-navy text-white"
                    : "text-navy hover:bg-surface"
                }`}
              >
                Table
              </button>
            </div>
            <Link href="/scrape">
              <Button>Add Job</Button>
            </Link>
          </div>
        </div>

        {/* Ready to Apply section */}
        {readyJobs.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-heading text-xl font-bold text-navy flex items-center gap-2">
                <Icon name="check_circle" size={18} />
                Ready to Apply ({readyJobs.length})
              </h3>
            </div>
            <div className="flex gap-4 overflow-x-auto pb-4">
              {readyJobs.slice(0, 5).map((job) => (
                <button
                  key={job.id}
                  onClick={() => handleStatusChange(job.id, "applied")}
                  className="min-w-[200px] bg-card p-4 border border-border-light hover:border-navy transition-sharp text-left"
                >
                  <p className="font-heading font-bold text-sm">{job.title}</p>
                  <p className="text-[9px] text-muted uppercase tracking-widest mt-1">
                    {job.company || "Unknown"}
                  </p>
                  <p className="text-[8px] text-muted uppercase tracking-widest mt-3 flex items-center gap-1">
                    <Icon name="arrow_forward" size={10} /> Click to mark applied
                  </p>
                </button>
              ))}
            </div>
          </section>
        )}

        {/* Kanban Board */}
        {view === "kanban" ? (
          <KanbanBoard
            jobs={trackerJobs}
            onStatusChange={handleStatusChange}
          />
        ) : (
          /* Table View - simple list */
          <div className="bg-card border border-navy">
            <div className="px-8 py-4 border-b border-navy">
              <h3 className="font-heading font-bold uppercase tracking-widest">
                All Tracked Applications
              </h3>
            </div>
            <table className="w-full text-left">
              <thead className="bg-surface text-[9px] uppercase tracking-[0.2em] text-muted border-b border-navy">
                <tr>
                  <th className="px-8 py-4 font-bold">ID</th>
                  <th className="px-8 py-4 font-bold">Title</th>
                  <th className="px-8 py-4 font-bold">Company</th>
                  <th className="px-8 py-4 font-bold">Status</th>
                  <th className="px-8 py-4 font-bold">Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-light">
                {trackerJobs.map((job) => (
                  <tr key={job.id} className="hover:bg-surface transition-sharp">
                    <td className="px-8 py-4 font-mono text-[10px]">JH-{job.id}</td>
                    <td className="px-8 py-4 font-heading font-bold">{job.title}</td>
                    <td className="px-8 py-4 text-[10px] text-muted uppercase tracking-widest">
                      {job.company}
                    </td>
                    <td className="px-8 py-4">
                      <span className="text-[9px] font-bold px-2 py-0.5 border border-navy uppercase">
                        {job.status}
                      </span>
                    </td>
                    <td className="px-8 py-4 font-mono text-sm font-bold">
                      {job.fit_score ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {trackerJobs.length === 0 && (
              <div className="px-8 py-16 text-center">
                <p className="text-muted text-[11px] uppercase tracking-widest">
                  No tracked applications. Tailor resumes and mark jobs as applied to populate.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Bottom: Terminal + Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 pt-12 border-t border-navy">
          <div className="lg:col-span-2">
            <TerminalFeed
              entries={entries}
              title="Bounty Hunter Terminal Output"
              height="h-64"
              variant="dark"
            />
          </div>
          <div className="border border-navy bg-card p-8">
            <h3 className="font-heading font-bold text-navy mb-8 uppercase tracking-widest text-sm border-b border-navy pb-2">
              Division Metrics
            </h3>
            <div className="space-y-8">
              <div>
                <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest text-navy mb-2">
                  <span>Conversion</span>
                  <span>{conversionRate}%</span>
                </div>
                <div className="h-[2px] bg-surface-alt">
                  <div className="h-full bg-navy" style={{ width: `${conversionRate}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest text-navy mb-2">
                  <span>Response</span>
                  <span>{responseRate}%</span>
                </div>
                <div className="h-[2px] bg-surface-alt">
                  <div className="h-full bg-navy" style={{ width: `${responseRate}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest text-navy mb-2">
                  <span>Health</span>
                  <span>OPTIMAL</span>
                </div>
                <div className="h-[2px] bg-surface-alt">
                  <div className="h-full bg-navy w-[88%]" />
                </div>
              </div>
              <div className="pt-4 mt-8 border-t border-border-light">
                <p className="text-[8px] text-muted font-heading italic text-center">
                  &quot;Keep hunting. The best offer is the next one.&quot;
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
