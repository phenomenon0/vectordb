"use client";

import { useState, useCallback } from "react";
import { Topbar } from "@/components/layout/topbar";
import { useJobs, useStats, useMutation, useTerminalLog } from "@/lib/hooks";
import { SourcesPanel } from "@/components/scrape/sources-panel";
import { PrecisionFilters } from "@/components/scrape/precision-filters";
import { PipelineTable } from "@/components/scrape/pipeline-table";
import { TerminalFeed, type LogEntry } from "@/components/ui/terminal-feed";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";

export default function ScrapePage() {
  const { jobs, refresh: refreshJobs } = useJobs();
  const { stats, refresh: refreshStats } = useStats();
  const { entries: logEntries, addEntry } = useTerminalLog(stats);
  const scrapeMutation = useMutation("/api/scrape");
  const filterMutation = useMutation("/api/filter");

  // Form state
  const [role, setRole] = useState("");
  const [location, setLocation] = useState("");
  const [count, setCount] = useState(50);
  const [activeSites, setActiveSites] = useState(["linkedin", "indeed"]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [scrapeProgress, setScrapeProgress] = useState<number | null>(null);

  // Terminal entries for command output
  const [commandEntries, setCommandEntries] = useState<LogEntry[]>([]);

  const toggleSite = (site: string) => {
    setActiveSites((prev) =>
      prev.includes(site) ? prev.filter((s) => s !== site) : [...prev, site]
    );
  };

  const handleScrape = useCallback(async () => {
    if (!role.trim()) return;
    setScrapeProgress(30);
    const now = () => new Date().toLocaleTimeString("en-US", { hour12: false });
    setCommandEntries((prev) => [
      ...prev,
      { timestamp: now(), level: "info", message: `REQUEST: SCRAPE "${role.toUpperCase()}" FROM ${activeSites.join(", ").toUpperCase()}` },
    ]);

    try {
      setScrapeProgress(60);
      await scrapeMutation.execute({
        role,
        location: location || undefined,
        count,
        sites: activeSites.join(","),
      });
      setScrapeProgress(100);
      setCommandEntries((prev) => [
        ...prev,
        { timestamp: now(), level: "success", message: `SCRAPE COMPLETE. REFRESHING PIPELINE...` },
      ]);
      await Promise.all([refreshJobs(), refreshStats()]);
    } catch (e) {
      setCommandEntries((prev) => [
        ...prev,
        { timestamp: now(), level: "error", message: `SCRAPE FAILED: ${e}` },
      ]);
    } finally {
      setScrapeProgress(null);
    }
  }, [role, location, count, activeSites, scrapeMutation, refreshJobs, refreshStats]);

  const handleFilter = useCallback(async () => {
    const now = () => new Date().toLocaleTimeString("en-US", { hour12: false });
    setCommandEntries((prev) => [
      ...prev,
      { timestamp: now(), level: "info", message: "INITIATING SCORER_V4 PIPELINE..." },
    ]);
    try {
      await filterMutation.execute({});
      setCommandEntries((prev) => [
        ...prev,
        { timestamp: now(), level: "success", message: "SCORING COMPLETE. PIPELINE UPDATED." },
      ]);
      await Promise.all([refreshJobs(), refreshStats()]);
    } catch (e) {
      setCommandEntries((prev) => [
        ...prev,
        { timestamp: now(), level: "error", message: `SCORING FAILED: ${e}` },
      ]);
    }
  }, [filterMutation, refreshJobs, refreshStats]);

  return (
    <>
      <Topbar title="Find Jobs" subtitle="Search across job boards and filter the best matches" />

      <div className="flex-1 overflow-y-auto p-10 space-y-12 custom-scroll">
        {/* Hero Header */}
        <section className="border-b border-navy pb-6">
          <h1 className="font-heading text-4xl font-black text-navy tracking-tighter uppercase">
            Find Jobs
          </h1>
          <p className="text-muted mt-2 font-sans text-xs uppercase tracking-[0.2em]">
            Search job boards, filter by relevance, and build your pipeline
          </p>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          {/* Left: Sources + Filters */}
          <div className="lg:col-span-4 space-y-10">
            <SourcesPanel
              activeSites={activeSites}
              onToggle={toggleSite}
              scrapeProgress={scrapeProgress}
            />
            <PrecisionFilters
              role={role}
              onRoleChange={setRole}
              location={location}
              onLocationChange={setLocation}
              count={count}
              onCountChange={setCount}
            />

            {/* Action buttons */}
            <div className="space-y-3">
              <Button
                onClick={handleScrape}
                disabled={scrapeMutation.loading || !role.trim()}
                className="w-full flex items-center justify-center gap-2"
              >
                {scrapeMutation.loading ? (
                  <span className="animate-pulse">SCRAPING...</span>
                ) : (
                  <>
                    <Icon name="play_circle" size={14} className="text-white" />
                    INITIATE SCRAPE
                  </>
                )}
              </Button>
              <Button
                variant="secondary"
                onClick={handleFilter}
                disabled={filterMutation.loading}
                className="w-full flex items-center justify-center gap-2"
              >
                {filterMutation.loading ? (
                  <span className="animate-pulse">SCORING...</span>
                ) : (
                  <>
                    <Icon name="filter_alt" size={14} />
                    SCORE PIPELINE
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Right: Pipeline Table */}
          <div className="lg:col-span-8">
            <PipelineTable
              jobs={jobs}
              globalFilter={globalFilter}
              onGlobalFilterChange={setGlobalFilter}
            />

            {/* Terminal Logs */}
            <div className="mt-10">
              <TerminalFeed
                entries={[...logEntries, ...commandEntries]}
                title="TELEMETRY_LOGS // SCRAPE_PIPELINE"
                height="h-32"
                variant="dark"
              />
            </div>
          </div>
        </div>
      </div>

      {/* FAB */}
      <button
        onClick={handleScrape}
        disabled={scrapeMutation.loading || !role.trim()}
        className="fixed bottom-10 right-10 w-16 h-16 bg-navy text-white border border-navy flex items-center justify-center hover:bg-primary transition-sharp z-10 disabled:opacity-30"
      >
        <Icon name="rocket_launch" size={28} className="text-white" />
      </button>
    </>
  );
}
