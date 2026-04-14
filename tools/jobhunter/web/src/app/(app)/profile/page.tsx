"use client";

import { useState, useEffect } from "react";
import { Topbar } from "@/components/layout/topbar";
import { useProfile, type UserProfile } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";

const EXPERIENCE_LEVELS = [
  { value: "student", label: "Student / New Grad" },
  { value: "early", label: "Early Career (1-3 yrs)" },
  { value: "mid", label: "Mid-Level (4-7 yrs)" },
  { value: "senior", label: "Senior (8+ yrs)" },
  { value: "lead", label: "Lead / Staff" },
  { value: "executive", label: "Director+" },
];

export default function ProfilePage() {
  const { profile, loading, save } = useProfile();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Local form state
  const [name, setName] = useState("");
  const [headline, setHeadline] = useState("");
  const [targetRoles, setTargetRoles] = useState("");
  const [experience, setExperience] = useState("");
  const [location, setLocation] = useState("");
  const [salary, setSalary] = useState("");
  const [skills, setSkills] = useState("");
  const [linkedin, setLinkedin] = useState("");
  const [portfolio, setPortfolio] = useState("");

  // Populate from profile when loaded
  useEffect(() => {
    if (profile) {
      setName(profile.name || "");
      setHeadline(profile.headline || "");
      setTargetRoles(profile.targetRoles?.join(", ") || "");
      setExperience(profile.experienceLevel || "");
      setLocation(profile.locationPreference || "");
      setSalary(profile.salaryExpectation || "");
      setSkills(profile.skills?.join(", ") || "");
      setLinkedin(profile.linkedinUrl || "");
      setPortfolio(profile.portfolioUrl || "");
    }
  }, [profile]);

  async function handleSave() {
    setSaving(true);
    setSaved(false);
    await save({
      name,
      headline,
      targetRoles: targetRoles.split(",").map((r) => r.trim()).filter(Boolean),
      experienceLevel: experience,
      locationPreference: location,
      salaryExpectation: salary,
      skills: skills.split(",").map((s) => s.trim()).filter(Boolean),
      linkedinUrl: linkedin,
      portfolioUrl: portfolio,
    });
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  if (loading) {
    return (
      <>
        <Topbar title="Profile" />
        <div className="p-12 max-w-2xl">
          <div className="space-y-6">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-12 bg-surface-alt animate-pulse" />
            ))}
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Topbar title="Your Profile" subtitle="Who you are and what you're looking for" />

      <div className="flex-1 overflow-y-auto custom-scroll p-8 md:p-12">
        <div className="max-w-2xl space-y-12">
          {/* Header */}
          <section className="border-b border-navy pb-6">
            <h1 className="font-heading text-3xl font-bold tracking-tight">
              Your Profile
            </h1>
            <p className="text-muted text-[11px] mt-2 leading-relaxed">
              This information powers your job matching and helps AI tailor your
              applications. The more you fill in, the better your results.
            </p>
          </section>

          {/* Identity */}
          <section>
            <h2 className="font-heading text-xl font-bold mb-6 flex items-center gap-2">
              <Icon name="person" size={20} />
              Identity
            </h2>
            <div className="space-y-6 bg-card p-8 border border-border-light">
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Full Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Alex Rivera"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
              </div>
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Headline
                </label>
                <input
                  type="text"
                  value={headline}
                  onChange={(e) => setHeadline(e.target.value)}
                  placeholder="Full-Stack Engineer passionate about distributed systems"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
                <p className="text-[9px] text-muted/60 mt-2">
                  A one-liner that describes you professionally
                </p>
              </div>
            </div>
          </section>

          {/* Job Preferences */}
          <section>
            <h2 className="font-heading text-xl font-bold mb-6 flex items-center gap-2">
              <Icon name="target" size={20} />
              What You&apos;re Looking For
            </h2>
            <div className="space-y-6 bg-card p-8 border border-border-light">
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Target Roles
                </label>
                <input
                  type="text"
                  value={targetRoles}
                  onChange={(e) => setTargetRoles(e.target.value)}
                  placeholder="Software Engineer, ML Engineer, Backend Developer"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
                <p className="text-[9px] text-muted/60 mt-2">
                  Separate multiple roles with commas
                </p>
              </div>

              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Experience Level
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {EXPERIENCE_LEVELS.map((lvl) => (
                    <button
                      key={lvl.value}
                      type="button"
                      onClick={() => setExperience(lvl.value)}
                      className={`p-3 text-[10px] font-bold uppercase tracking-widest transition-sharp border text-center ${
                        experience === lvl.value
                          ? "bg-navy text-white border-navy"
                          : "bg-surface border-border-light hover:border-navy"
                      }`}
                    >
                      {lvl.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                    Preferred Location
                  </label>
                  <input
                    type="text"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder="Remote, London, NYC"
                    className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                  />
                </div>
                <div>
                  <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                    Salary Expectation
                  </label>
                  <input
                    type="text"
                    value={salary}
                    onChange={(e) => setSalary(e.target.value)}
                    placeholder="$120k - $180k"
                    className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Key Skills
                </label>
                <input
                  type="text"
                  value={skills}
                  onChange={(e) => setSkills(e.target.value)}
                  placeholder="Python, Go, React, Kubernetes, Distributed Systems"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
                <p className="text-[9px] text-muted/60 mt-2">
                  Separate skills with commas — these help the AI match and tailor
                </p>
              </div>
            </div>
          </section>

          {/* Links */}
          <section>
            <h2 className="font-heading text-xl font-bold mb-6 flex items-center gap-2">
              <Icon name="link" size={20} />
              Links
            </h2>
            <div className="space-y-6 bg-card p-8 border border-border-light">
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  LinkedIn
                </label>
                <input
                  type="url"
                  value={linkedin}
                  onChange={(e) => setLinkedin(e.target.value)}
                  placeholder="https://linkedin.com/in/yourprofile"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
              </div>
              <div>
                <label className="block text-[9px] font-bold text-muted uppercase tracking-[0.2em] mb-3">
                  Portfolio / Website
                </label>
                <input
                  type="url"
                  value={portfolio}
                  onChange={(e) => setPortfolio(e.target.value)}
                  placeholder="https://yoursite.com"
                  className="w-full bg-surface border-b-2 border-border-light focus:border-navy text-sm py-3 px-1 text-navy placeholder-muted/40"
                />
              </div>
            </div>
          </section>

          {/* Save */}
          <div className="flex items-center gap-4 pb-12">
            <Button onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Save Profile"}
            </Button>
            {saved && (
              <span className="text-[10px] font-bold text-navy flex items-center gap-1">
                <Icon name="check_circle" size={14} /> Saved
              </span>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
