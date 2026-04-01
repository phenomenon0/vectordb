"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Topbar } from "@/components/layout/topbar";
import { useJobs, useJobDetail, useConfig, useMutation } from "@/lib/hooks";
import { JobPicker } from "@/components/tailor/job-picker";
import { ReferencePane } from "@/components/tailor/reference-pane";
import { EditorPane } from "@/components/tailor/editor-pane";
import { KeywordMetrics } from "@/components/tailor/keyword-metrics";
import { Icon } from "@/components/ui/icon";
import { api } from "@/lib/api";

export default function TailorPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { jobs } = useJobs();
  const { config } = useConfig();
  const tailorMutation = useMutation("/api/tailor");

  const initialId = searchParams.get("id");
  const [selectedId, setSelectedId] = useState<number | null>(
    initialId ? parseInt(initialId, 10) : null
  );

  const { job, loading: jobLoading } = useJobDetail(selectedId);

  // Filter to relevant + tailored jobs (ones worth tailoring)
  const tailorableJobs = useMemo(
    () =>
      jobs.filter(
        (j) => j.status === "relevant" || j.status === "tailored"
      ),
    [jobs]
  );

  // Auto-select first job if none selected
  useEffect(() => {
    if (!selectedId && tailorableJobs.length > 0) {
      setSelectedId(tailorableJobs[0].id);
    }
  }, [selectedId, tailorableJobs]);

  const handleRegenerate = useCallback(async () => {
    try {
      await tailorMutation.execute({});
      // Refresh the job detail
      if (selectedId) {
        // Force re-fetch by toggling
        const id = selectedId;
        setSelectedId(null);
        setTimeout(() => setSelectedId(id), 100);
      }
    } catch {}
  }, [tailorMutation, selectedId]);

  const handleCommit = useCallback(async () => {
    if (!selectedId) return;
    try {
      await fetch(api(`/api/jobs/${selectedId}/status`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "applied" }),
      });
      router.push("/tracker");
    } catch (e) {
      console.error("Failed to commit:", e);
    }
  }, [selectedId, router]);

  return (
    <>
      <Topbar title="Tailor Workspace" />

      <main className="flex-1 overflow-hidden flex flex-col lg:flex-row">
        {/* Left Pane: Source Context */}
        <section className="lg:w-2/5 border-r border-navy/10 flex flex-col bg-surface">
          <div className="p-6 border-b border-navy/10 flex items-center justify-between bg-surface">
            <div className="flex items-center gap-3">
              <Icon name="description" size={18} />
              <h2 className="font-heading font-bold text-navy uppercase tracking-widest text-sm">
                Reference Data
              </h2>
            </div>
            <span className="text-[9px] border border-navy px-2 py-1 font-bold text-navy uppercase tracking-widest">
              Source Lock
            </span>
          </div>

          {/* Job Picker */}
          <div className="p-4 border-b border-navy/10">
            <JobPicker
              jobs={tailorableJobs}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </div>

          {/* Job Details + Resume */}
          <ReferencePane
            job={job}
            loading={jobLoading}
            resumePath={config?.resume_path}
          />
        </section>

        {/* Right Pane: AI Editor */}
        <div className="flex-1 flex">
          <EditorPane
            job={job}
            onRegenerate={handleRegenerate}
            onCommit={handleCommit}
            regenerating={tailorMutation.loading}
          />

          {/* Keyword Metrics Sidebar */}
          <div className="w-44 hidden xl:block p-4">
            <KeywordMetrics job={job} />
          </div>
        </div>
      </main>
    </>
  );
}
