"use client";

import { useState, useRef } from "react";
import { Icon } from "./ui/icon";
import { Button } from "./ui/button";
import { api } from "@/lib/api";

interface OnboardingProps {
  onComplete: () => void;
}

const EXPERIENCE_LEVELS = [
  { value: "student", label: "Student / New Grad", desc: "Just starting out" },
  { value: "early", label: "Early Career", desc: "1-3 years experience" },
  { value: "mid", label: "Mid-Level", desc: "4-7 years experience" },
  { value: "senior", label: "Senior", desc: "8+ years experience" },
  { value: "lead", label: "Lead / Staff", desc: "Technical leadership" },
  { value: "executive", label: "Director+", desc: "Executive level" },
];

export function Onboarding({ onComplete }: OnboardingProps) {
  const [step, setStep] = useState(1);

  // Step 1: About You
  const [name, setName] = useState("");
  const [targetRole, setTargetRole] = useState("");
  const [experience, setExperience] = useState("");
  const [location, setLocation] = useState("");

  // Step 2: Resume
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadDone, setUploadDone] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // Step 3: finishing
  const [saving, setSaving] = useState(false);

  async function handleUpload() {
    if (!resumeFile) return;
    setUploading(true);
    setUploadError("");
    try {
      const form = new FormData();
      form.append("resume", resumeFile);
      const res = await fetch(api("/api/upload"), { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setUploadDone(true);
    } catch (e) {
      setUploadError(String(e));
    } finally {
      setUploading(false);
    }
  }

  async function handleFinish() {
    setSaving(true);
    try {
      // Save profile
      await fetch(api("/api/profile"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          targetRoles: targetRole ? targetRole.split(",").map((r) => r.trim()) : [],
          experienceLevel: experience,
          locationPreference: location,
        }),
      });

      // Write basic config (model defaults, no API key required)
      await fetch(api("/api/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-haiku-4-5-20251001" }),
      });

      onComplete();
    } catch {
      onComplete();
    } finally {
      setSaving(false);
    }
  }

  const canProceedStep1 = name.trim().length > 0;

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-surface">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="font-heading text-3xl font-bold text-navy mb-2">
            Bounty Hunter
          </h1>
          <p className="text-[11px] text-muted">
            Your AI-powered job search companion
          </p>
          <div className="h-px bg-border-light my-6 w-24 mx-auto" />
        </div>

        {/* Progress */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {[1, 2, 3].map((s) => (
            <div key={s} className="flex items-center gap-2">
              <div
                className={`w-8 h-8 flex items-center justify-center text-[10px] font-bold transition-sharp border ${
                  step > s
                    ? "bg-primary text-white border-primary"
                    : step === s
                      ? "bg-primary text-white border-primary"
                      : "bg-card text-muted border-border-light"
                }`}
              >
                {step > s ? (
                  <Icon name="check" size={14} className="text-white" />
                ) : (
                  s
                )}
              </div>
              {s < 3 && (
                <div
                  className={`w-12 h-px ${step > s ? "bg-primary" : "bg-surface-alt"}`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Step labels */}
        <div className="flex justify-center gap-6 mb-8">
          {["About You", "Resume", "Ready"].map((label, i) => (
            <span
              key={label}
              className={`text-[9px] uppercase tracking-widest font-bold ${
                step === i + 1 ? "text-navy" : "text-muted/40"
              }`}
            >
              {label}
            </span>
          ))}
        </div>

        {/* ─── STEP 1: About You ─── */}
        {step === 1 && (
          <div className="bg-card border border-border-light p-8 shadow-sm">
            <div className="mb-8">
              <h2 className="font-heading text-xl font-semibold text-navy mb-2">
                Let&apos;s get to know you
              </h2>
              <p className="text-[11px] text-muted leading-relaxed">
                We&apos;ll use this to find the right jobs and tailor your applications.
                You can always update this later.
              </p>
            </div>

            <div className="space-y-6">
              {/* Name */}
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Your Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Alex Rivera"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-primary text-sm py-3 px-1 text-navy placeholder-muted/40"
                  autoFocus
                />
              </div>

              {/* Target Role */}
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  What roles are you looking for?
                </label>
                <input
                  type="text"
                  value={targetRole}
                  onChange={(e) => setTargetRole(e.target.value)}
                  placeholder="e.g. Software Engineer, ML Engineer"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-primary text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
                <p className="text-[9px] text-muted/60 mt-2">
                  Separate multiple roles with commas
                </p>
              </div>

              {/* Experience Level */}
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Experience Level
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {EXPERIENCE_LEVELS.map((lvl) => (
                    <button
                      key={lvl.value}
                      type="button"
                      onClick={() => setExperience(lvl.value)}
                      className={`p-3 text-left transition-sharp border ${
                        experience === lvl.value
                          ? "bg-primary text-white border-primary"
                          : "bg-surface border-border-light hover:border-primary"
                      }`}
                    >
                      <span className={`text-[10px] font-bold block ${experience === lvl.value ? "text-white" : "text-navy"}`}>
                        {lvl.label}
                      </span>
                      <span className={`text-[9px] ${experience === lvl.value ? "text-white/60" : "text-muted"}`}>
                        {lvl.desc}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Location */}
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Preferred Location
                </label>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g. Remote, San Francisco, London"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-primary text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
              </div>
            </div>

            <div className="flex justify-end mt-8">
              <Button onClick={() => setStep(2)} disabled={!canProceedStep1}>
                Continue
                <Icon name="chevron_right" size={14} className="text-white ml-1" />
              </Button>
            </div>
          </div>
        )}

        {/* ─── STEP 2: Resume ─── */}
        {step === 2 && (
          <div className="bg-card border border-border-light p-8 shadow-sm">
            <div className="mb-8">
              <h2 className="font-heading text-xl font-semibold text-navy mb-2">
                Upload your resume
              </h2>
              <p className="text-[11px] text-muted leading-relaxed">
                This helps us match you with the right jobs and generate tailored
                cover letters. You can skip this and add it later in Settings.
              </p>
            </div>

            {/* Drop zone */}
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className={`w-full border-2 border-dashed p-10 text-center cursor-pointer transition-sharp ${
                resumeFile
                  ? "border-primary bg-surface"
                  : "border-border-light hover:border-primary hover:bg-surface"
              }`}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.md,.txt"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    setResumeFile(f);
                    setUploadDone(false);
                  }
                }}
              />
              {resumeFile ? (
                <div>
                  <Icon name="check_circle" size={36} className="text-navy mb-3" />
                  <p className="font-bold text-sm">{resumeFile.name}</p>
                  <p className="text-[9px] text-muted mt-2">
                    {(resumeFile.size / 1024).toFixed(1)} KB — Click to change
                  </p>
                </div>
              ) : (
                <div>
                  <Icon name="upload_file" size={36} className="text-border mb-3" />
                  <p className="text-sm text-muted mb-1">
                    Click to select your resume
                  </p>
                  <p className="text-[9px] text-muted/50">
                    PDF, Markdown, or plain text
                  </p>
                </div>
              )}
            </button>

            {uploadError && (
              <p className="text-error text-[10px] mt-3">{uploadError}</p>
            )}

            <div className="flex justify-between mt-8">
              <button
                onClick={() => setStep(1)}
                className="text-[10px] text-muted hover:text-navy uppercase tracking-widest font-bold transition-sharp"
              >
                Back
              </button>
              <div className="flex gap-3">
                {/* Skip option */}
                <Button variant="ghost" onClick={() => setStep(3)}>
                  Skip for now
                </Button>

                {!uploadDone && resumeFile ? (
                  <Button onClick={handleUpload} disabled={uploading}>
                    {uploading ? "Uploading..." : "Upload"}
                  </Button>
                ) : uploadDone ? (
                  <Button onClick={() => setStep(3)}>
                    Continue
                    <Icon name="chevron_right" size={14} className="text-white ml-1" />
                  </Button>
                ) : null}
              </div>
            </div>
          </div>
        )}

        {/* ─── STEP 3: You're All Set ─── */}
        {step === 3 && (
          <div className="bg-card border border-border-light p-8 shadow-sm">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-primary flex items-center justify-center mx-auto mb-6">
                <Icon name="check" size={32} className="text-white" />
              </div>
              <h2 className="font-heading text-xl font-semibold text-navy mb-2">
                You&apos;re all set, {name.split(" ")[0] || "there"}
              </h2>
              <p className="text-[11px] text-muted leading-relaxed">
                Here&apos;s what we&apos;ll help you do next.
              </p>
            </div>

            {/* Summary */}
            <div className="space-y-4 mb-8">
              <div className="flex items-start gap-4 p-4 bg-surface">
                <div className="w-8 h-8 bg-primary flex items-center justify-center shrink-0">
                  <Icon name="search" size={16} className="text-white" />
                </div>
                <div>
                  <p className="font-bold text-[11px] uppercase tracking-widest">
                    Find Jobs
                  </p>
                  <p className="text-[10px] text-muted mt-1">
                    We&apos;ll scrape LinkedIn, Indeed & more for{" "}
                    <span className="font-bold text-navy">
                      {targetRole || "your target roles"}
                    </span>{" "}
                    {location && (
                      <>
                        in <span className="font-bold text-navy">{location}</span>
                      </>
                    )}
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-surface">
                <div className="w-8 h-8 bg-primary flex items-center justify-center shrink-0">
                  <Icon name="auto_fix_high" size={16} className="text-white" />
                </div>
                <div>
                  <p className="font-bold text-[11px] uppercase tracking-widest">
                    Tailor Applications
                  </p>
                  <p className="text-[10px] text-muted mt-1">
                    AI scores each job against your profile and generates tailored
                    cover letters and resumes
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-surface">
                <div className="w-8 h-8 bg-primary flex items-center justify-center shrink-0">
                  <Icon name="analytics" size={16} className="text-white" />
                </div>
                <div>
                  <p className="font-bold text-[11px] uppercase tracking-widest">
                    Track Progress
                  </p>
                  <p className="text-[10px] text-muted mt-1">
                    Kanban board to track applications from applied through to offer
                  </p>
                </div>
              </div>
            </div>

            <p className="text-[9px] text-muted text-center mb-6">
              You can add your API key and adjust preferences anytime in{" "}
              <span className="font-bold">Settings</span>
            </p>

            <Button
              onClick={handleFinish}
              disabled={saving}
              className="w-full flex items-center justify-center"
            >
              {saving ? "Setting things up..." : "Go to Dashboard"}
              {!saving && <Icon name="arrow_forward" size={14} className="text-white ml-2" />}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
