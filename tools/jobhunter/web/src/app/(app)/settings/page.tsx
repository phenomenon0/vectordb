"use client";

import { useState, useEffect, useRef } from "react";
import { Topbar } from "@/components/layout/topbar";
import { useConfig } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const { config, loading, refresh } = useConfig();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Form state
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("claude-haiku-4-5-20251001");

  // Resume upload
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (config) {
      setModel(config.model || "claude-haiku-4-5-20251001");
    }
  }, [config]);

  async function handleSaveConfig() {
    setSaving(true);
    setSaved(false);
    try {
      await fetch(api("/api/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, apiKey: apiKey || undefined }),
      });
      await refresh();
      setSaved(true);
      setApiKey(""); // Clear for security
      setTimeout(() => setSaved(false), 3000);
    } catch {}
    setSaving(false);
  }

  async function handleUploadResume() {
    if (!resumeFile) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const form = new FormData();
      form.append("resume", resumeFile);
      const res = await fetch(api("/api/upload"), { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setUploadMsg("Resume uploaded successfully");
      setResumeFile(null);
      await refresh();
    } catch (e) {
      setUploadMsg(`Error: ${e}`);
    }
    setUploading(false);
  }

  if (loading) {
    return (
      <>
        <Topbar title="Settings" />
        <div className="p-12 max-w-2xl">
          <div className="space-y-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-12 bg-surface-alt animate-pulse" />
            ))}
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Topbar title="Settings" subtitle="Configure your job hunting toolkit" />

      <div className="flex-1 overflow-y-auto custom-scroll p-8 md:p-12">
        <div className="max-w-2xl space-y-12">
          {/* Header */}
          <section className="border-b border-navy pb-6">
            <h1 className="font-heading text-3xl font-bold tracking-tight">
              Settings
            </h1>
            <p className="text-muted text-[11px] mt-2 leading-relaxed">
              API keys, model preferences, and resume management.
              These power the AI features behind the scenes.
            </p>
          </section>

          {/* AI Configuration */}
          <section>
            <h2 className="font-heading text-xl font-bold mb-6 flex items-center gap-2">
              <Icon name="smart_toy" size={20} />
              AI Configuration
            </h2>
            <div className="space-y-6 bg-card p-8 border border-border-light">
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Anthropic API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={config?.hasApiKey ? "••••••••••••  (key saved)" : "sk-ant-api03-..."}
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 font-mono text-navy placeholder-muted/40"
                />
                <p className="text-[9px] text-muted mt-2">
                  {config?.hasApiKey ? (
                    <span className="flex items-center gap-1">
                      <Icon name="check_circle" size={10} /> API key is configured. Enter a new one to replace it.
                    </span>
                  ) : (
                    <>
                      Required for AI scoring and tailoring. Get one at{" "}
                      <span className="font-bold">console.anthropic.com</span>
                      {" "}— or set ANTHROPIC_API_KEY in your shell environment.
                    </>
                  )}
                </p>
              </div>

              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  AI Model
                </label>
                <div className="space-y-2">
                  {[
                    {
                      value: "claude-haiku-4-5-20251001",
                      label: "Claude Haiku 4.5",
                      desc: "Fast & affordable — great for scoring lots of jobs",
                    },
                    {
                      value: "claude-sonnet-4-20250514",
                      label: "Claude Sonnet 4",
                      desc: "More nuanced — better for tailoring cover letters",
                    },
                  ].map((m) => (
                    <button
                      key={m.value}
                      type="button"
                      onClick={() => setModel(m.value)}
                      className={`w-full p-4 text-left transition-sharp border flex items-center justify-between ${
                        model === m.value
                          ? "bg-navy text-white border-navy"
                          : "bg-surface border-border-light hover:border-navy"
                      }`}
                    >
                      <div>
                        <span className={`font-bold text-[11px] block ${model === m.value ? "text-white" : "text-navy"}`}>
                          {m.label}
                        </span>
                        <span className={`text-[9px] ${model === m.value ? "text-white/60" : "text-muted"}`}>
                          {m.desc}
                        </span>
                      </div>
                      {model === m.value && (
                        <Icon name="check" size={16} className="text-white" />
                      )}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-4 pt-4">
                <Button onClick={handleSaveConfig} disabled={saving}>
                  {saving ? "Saving..." : "Save AI Settings"}
                </Button>
                {saved && (
                  <span className="text-[10px] font-bold text-navy flex items-center gap-1">
                    <Icon name="check_circle" size={14} /> Saved
                  </span>
                )}
              </div>
            </div>
          </section>

          {/* Resume Management */}
          <section>
            <h2 className="font-heading text-xl font-bold mb-6 flex items-center gap-2">
              <Icon name="description" size={20} />
              Resume
            </h2>
            <div className="space-y-6 bg-card p-8 border border-border-light">
              {/* Current resume */}
              {config?.resumeName ? (
                <div className="flex items-center gap-4 p-4 bg-surface">
                  <div className="w-10 h-10 bg-navy flex items-center justify-center">
                    <Icon name="description" size={20} className="text-white" />
                  </div>
                  <div>
                    <p className="font-bold text-[11px]">{config.resumeName}</p>
                    <p className="text-[9px] text-muted">Current resume on file</p>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-surface text-center">
                  <p className="text-[10px] text-muted">
                    No resume uploaded yet. Upload one to enable AI tailoring.
                  </p>
                </div>
              )}

              {/* Upload new */}
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  {config?.resumeName ? "Replace Resume" : "Upload Resume"}
                </label>
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  className={`w-full border-2 border-dashed p-6 text-center cursor-pointer transition-sharp ${
                    resumeFile
                      ? "border-navy bg-surface"
                      : "border-border-light hover:border-navy"
                  }`}
                >
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".pdf,.md,.txt"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) setResumeFile(f);
                    }}
                  />
                  {resumeFile ? (
                    <p className="text-sm font-bold">{resumeFile.name}</p>
                  ) : (
                    <p className="text-[10px] text-muted">
                      Click to select — PDF, Markdown, or plain text
                    </p>
                  )}
                </button>

                {resumeFile && (
                  <div className="mt-4">
                    <Button onClick={handleUploadResume} disabled={uploading}>
                      {uploading ? "Uploading..." : "Upload Resume"}
                    </Button>
                  </div>
                )}

                {uploadMsg && (
                  <p className="text-[10px] mt-3 font-bold">{uploadMsg}</p>
                )}
              </div>
            </div>
          </section>

          {/* About */}
          <section className="pb-12">
            <div className="bg-surface p-8 text-center">
              <h3 className="font-heading text-lg font-bold mb-2">
                Bounty Hunter
              </h3>
              <p className="text-[9px] text-muted uppercase tracking-widest">
                AI-powered job hunting — scrape, score, tailor, track
              </p>
              <p className="text-[8px] text-muted mt-4 font-heading italic">
                &quot;Keep hunting. The best offer is the next one.&quot;
              </p>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
