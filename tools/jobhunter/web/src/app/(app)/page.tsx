"use client";

import { useMemo } from "react";
import { Topbar } from "@/components/layout/topbar";
import { useStats, useJobs, useTerminalLog, useProfile } from "@/lib/hooks";
import { StatCard } from "@/components/ui/stat-card";
import { TerminalFeed } from "@/components/ui/terminal-feed";
import { Icon } from "@/components/ui/icon";
import Link from "next/link";

export default function DashboardPage() {
  const { stats } = useStats(5000);
  const { jobs } = useJobs();
  const { entries } = useTerminalLog(stats);
  const { profile } = useProfile();
  const firstName = profile?.name?.split(" ")[0];

  const topLeads = useMemo(
    () =>
      jobs
        .filter((j) => j.fit_score !== null && j.fit_score >= 6)
        .sort((a, b) => (b.fit_score ?? 0) - (a.fit_score ?? 0))
        .slice(0, 3),
    [jobs]
  );

  const unscoredCount = useMemo(
    () => jobs.filter((j) => j.fit_score === null && j.status === "scraped").length,
    [jobs]
  );

  return (
    <>
      <Topbar title={firstName ? `Welcome back, ${firstName}` : "Dashboard"} />

      <div className="p-6 md:p-10 max-w-7xl mx-auto w-full space-y-10">
        {/* Stats */}
        <section>
          <h2 className="font-heading text-2xl font-semibold text-navy mb-6">
            Your Job Search at a Glance
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard icon="dataset" value={stats.total} label="Jobs Found" badge="LIVE" color="blue" progress={stats.total > 0 ? 75 : 0} />
            <StatCard icon="filter_alt" value={stats.scored} label="Scored" color="amber" progress={stats.total > 0 ? (stats.scored / stats.total) * 100 : 0} />
            <StatCard icon="auto_fix_high" value={stats.tailored} label="Tailored" color="green" progress={stats.total > 0 ? (stats.tailored / stats.total) * 100 : 0} />
            <StatCard icon="analytics" value={stats.relevant} label="Relevant" color="blue" progress={stats.total > 0 ? (stats.relevant / stats.total) * 100 : 0} />
          </div>
        </section>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Activity Feed */}
          <section className="lg:col-span-2 flex flex-col">
            <h2 className="font-heading text-lg font-semibold text-navy flex items-center gap-2 mb-4">
              <Icon name="terminal" size={20} className="text-primary" />
              Activity Feed
            </h2>
            <TerminalFeed entries={entries} title="BOUNTY_HUNTER" height="h-[360px]" />
          </section>

          {/* Quick Actions */}
          <section className="flex flex-col gap-4">
            <h2 className="font-heading text-lg font-semibold text-navy">
              Quick Actions
            </h2>

            <div className="bg-card p-5 border border-border-light space-y-3">
              {[
                { href: "/scrape", icon: "search", label: "Find New Jobs", desc: "Search across job boards" },
                { href: "/scrape", icon: "leaderboard", label: `Score Jobs (${stats.scored})`, desc: "AI-rank your matches" },
                { href: "/tracker", icon: "assignment", label: "View Tracker", desc: "Track your applications" },
              ].map((action) => (
                <Link
                  key={action.label}
                  href={action.href}
                  className="flex items-center gap-3 p-3 border border-border-light hover:border-primary hover:bg-surface-alt transition-sharp cursor-pointer"
                >
                  <div className="w-9 h-9 bg-surface-alt flex items-center justify-center text-primary">
                    <Icon name={action.icon} size={18} />
                  </div>
                  <div>
                    <span className="text-[12px] font-semibold text-navy block">
                      {action.label}
                    </span>
                    <span className="text-[10px] text-muted">{action.desc}</span>
                  </div>
                </Link>
              ))}
            </div>

            {/* Attention card */}
            {unscoredCount > 0 && (
              <div className="bg-primary text-white p-5 relative">
                <h3 className="font-heading text-sm font-semibold mb-1">
                  {unscoredCount} jobs need scoring
                </h3>
                <p className="text-[11px] leading-relaxed opacity-80 mb-4">
                  Run the AI scorer to find your best matches.
                </p>
                <Link
                  href="/scrape"
                  className="inline-flex items-center gap-1 text-white text-[11px] font-semibold hover:underline cursor-pointer"
                >
                  Score now <Icon name="arrow_forward" size={14} className="text-white" />
                </Link>
              </div>
            )}

            {/* System status */}
            <div className="bg-card p-4 border border-border-light flex items-center gap-3 mt-auto">
              <span className="w-2 h-2 bg-accent" />
              <span className="text-[10px] text-muted font-mono">
                System healthy — all services operational
              </span>
            </div>
          </section>
        </div>

        {/* Top Leads */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="font-heading text-lg font-semibold text-navy">
              Top Matches
            </h2>
            <Link href="/scrape" className="text-primary text-[11px] font-semibold hover:underline cursor-pointer">
              View all jobs
            </Link>
          </div>

          {topLeads.length === 0 ? (
            <div className="bg-card border border-border-light p-10 text-center">
              <Icon name="search" size={32} className="text-border mx-auto mb-3" />
              <p className="text-muted text-[12px]">
                No matches yet. Search for jobs and run the AI scorer to see your top matches here.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {topLeads.map((job) => (
                <Link
                  key={job.id}
                  href={`/tailor?id=${job.id}`}
                  className="bg-card p-6 border border-border-light border-l-3 border-l-primary hover:border-primary transition-sharp cursor-pointer"
                >
                  <div className="flex justify-between items-start mb-4">
                    <div className="w-9 h-9 bg-primary/10 text-primary flex items-center justify-center font-heading font-bold text-sm">
                      {(job.company || "?")[0].toUpperCase()}
                    </div>
                    <span className="font-mono text-[10px] font-semibold text-accent bg-accent/10 px-2 py-0.5">
                      {job.fit_score}/10
                    </span>
                  </div>
                  <h4 className="font-heading text-[15px] font-semibold text-navy mb-1">
                    {job.title}
                  </h4>
                  <p className="text-[11px] text-muted">
                    {job.company || "Unknown"} {job.location ? `· ${job.location}` : ""}
                  </p>
                  {job.salary && (
                    <p className="text-[11px] text-accent-dark font-semibold mt-2">
                      {job.salary}
                    </p>
                  )}
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  );
}
