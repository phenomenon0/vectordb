"use client";

import { useState } from "react";
import { Icon } from "@/components/ui/icon";
import { Button } from "@/components/ui/button";
import type { JobDetail } from "@/lib/types";

interface EditorPaneProps {
  job: JobDetail | null;
  onRegenerate: () => void;
  onCommit: () => void;
  regenerating: boolean;
}

export function EditorPane({ job, onRegenerate, onCommit, regenerating }: EditorPaneProps) {
  const [mode, setMode] = useState<"cover" | "resume">("cover");

  const handleCopy = () => {
    if (job?.resumeContent) {
      navigator.clipboard.writeText(job.resumeContent);
    }
  };

  return (
    <section className="flex-1 flex flex-col bg-card relative">
      {/* Top Action Bar */}
      <div className="h-16 border-b border-navy/10 flex items-center justify-between px-8 bg-surface">
        <div className="flex items-center gap-6">
          <button className="px-4 py-2 bg-navy text-white text-[10px] font-bold uppercase tracking-widest flex items-center gap-2">
            <Icon name="auto_fix_high" size={14} fill className="text-white" />
            AI Active
          </button>
          <div className="h-6 w-px bg-navy/10" />
          <div className="flex gap-2">
            <button className="p-2 hover:bg-surface-alt text-navy transition-sharp">
              <Icon name="format_bold" size={18} />
            </button>
            <button className="p-2 hover:bg-surface-alt text-navy transition-sharp">
              <Icon name="format_italic" size={18} />
            </button>
            <button className="p-2 hover:bg-surface-alt text-navy transition-sharp">
              <Icon name="format_list_bulleted" size={18} />
            </button>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={onRegenerate}
            disabled={regenerating}
            className="flex items-center gap-2 px-4 py-2 text-[10px] font-bold text-navy uppercase tracking-widest hover:bg-surface-alt transition-sharp disabled:opacity-40"
          >
            <Icon name="refresh" size={14} />
            {regenerating ? "Processing..." : "Regenerate"}
          </button>
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-4 py-2 text-[10px] font-bold text-navy uppercase tracking-widest hover:bg-surface-alt transition-sharp"
          >
            <Icon name="content_copy" size={14} />
            Copy
          </button>
          <button
            onClick={onCommit}
            className="flex items-center gap-2 px-6 py-2 bg-navy text-white font-bold text-[10px] uppercase tracking-[0.2em] transition-sharp hover:bg-primary"
          >
            <Icon name="save" size={14} className="text-white" />
            Commit
          </button>
        </div>
      </div>

      {/* Editor Canvas */}
      <div className="flex-1 overflow-y-auto custom-scroll p-12 flex flex-col items-center bg-surface">
        {/* Mode Selector */}
        <div className="flex bg-card p-0 border border-navy mb-12">
          <button
            onClick={() => setMode("cover")}
            className={`px-8 py-3 text-[10px] font-bold uppercase tracking-widest ${
              mode === "cover"
                ? "bg-navy text-white"
                : "text-navy hover:bg-surface transition-sharp"
            }`}
          >
            Tailored Cover Letter
          </button>
          <button
            onClick={() => setMode("resume")}
            className={`px-8 py-3 text-[10px] font-bold uppercase tracking-widest ${
              mode === "resume"
                ? "bg-navy text-white"
                : "text-navy hover:bg-surface transition-sharp"
            }`}
          >
            Optimized Resume
          </button>
        </div>

        {/* Editor Paper */}
        <div className="w-full max-w-2xl bg-card border border-navy/10 p-16 min-h-[600px] relative">
          {!job ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-24">
              <Icon name="edit_document" size={48} className="text-border mb-6" />
              <p className="text-muted text-[11px] uppercase tracking-widest">
                Select a job from the reference panel to begin tailoring
              </p>
            </div>
          ) : !job.resumeContent ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-24">
              <Icon name="auto_fix_high" size={48} className="text-border mb-6" />
              <p className="text-muted text-[11px] uppercase tracking-widest mb-6">
                No tailored content yet for this position
              </p>
              <Button onClick={onRegenerate} disabled={regenerating}>
                {regenerating ? "Generating..." : "Generate Tailored Content"}
              </Button>
            </div>
          ) : (
            <div className="space-y-6 text-navy leading-[1.8] text-xs">
              {job.resumeContent.split("\n").map((line, i) => (
                <p key={i} className={line.trim() === "" ? "h-4" : ""}>
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
