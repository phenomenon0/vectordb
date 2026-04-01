"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { JobRow, JobDetail, Stats } from "./types";
import { api } from "./api";

export function useStats(pollInterval?: number) {
  const [stats, setStats] = useState<Stats>({ total: 0, scored: 0, relevant: 0, tailored: 0 });
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(api("/api/stats"));
      if (res.ok) setStats(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh().finally(() => setLoading(false));
    if (pollInterval) {
      const id = setInterval(refresh, pollInterval);
      return () => clearInterval(id);
    }
  }, [refresh, pollInterval]);

  return { stats, loading, refresh };
}

export function useJobs() {
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(api("/api/jobs"));
      if (res.ok) setJobs(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  return { jobs, loading, refresh };
}

export function useJobDetail(id: number | null) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) {
      setJob(null);
      return;
    }
    setLoading(true);
    fetch(api(`/api/jobs/${id}`))
      .then((r) => r.json())
      .then(setJob)
      .catch(() => setJob(null))
      .finally(() => setLoading(false));
  }, [id]);

  return { job, loading };
}

export function useConfig() {
  const [config, setConfig] = useState<{
    isOnboarded: boolean;
    model?: string;
    resume_path?: string;
    resumePath?: string;
    resumeName?: string;
    resumeExists?: boolean;
    hasApiKey?: boolean;
    sites?: string[];
  } | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(api("/api/config"));
      if (res.ok) setConfig(await res.json());
    } catch {}
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  return { config, loading, refresh };
}

export interface UserProfile {
  name: string;
  headline: string;
  targetRoles: string[];
  experienceLevel: string;
  locationPreference: string;
  salaryExpectation: string;
  skills: string[];
  linkedinUrl: string;
  portfolioUrl: string;
  onboardedAt: string | null;
}

export function useProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(api("/api/profile"));
      if (res.ok) setProfile(await res.json());
    } catch {}
  }, []);

  const save = useCallback(async (data: Partial<UserProfile>) => {
    const res = await fetch(api("/api/profile"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await res.json();
    if (result.profile) setProfile(result.profile);
    return result;
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  return { profile, loading, refresh, save };
}

export function useMutation(endpoint: string) {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async (body?: Record<string, unknown>, method = "POST") => {
      setLoading(true);
      setOutput("");
      setError(null);
      try {
        const res = await fetch(api(endpoint), {
          method,
          headers: { "Content-Type": "application/json" },
          body: body ? JSON.stringify(body) : undefined,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        setOutput(data.output || JSON.stringify(data));
        return data;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setOutput(msg);
        throw e;
      } finally {
        setLoading(false);
      }
    },
    [endpoint]
  );

  return { execute, loading, output, error };
}

// Generates synthetic terminal log entries from stats changes
export function useTerminalLog(stats: Stats) {
  const [entries, setEntries] = useState<
    { timestamp: string; level: "info" | "warn" | "error" | "success"; message: string }[]
  >([]);
  const prevStats = useRef<Stats | null>(null);

  useEffect(() => {
    const now = new Date().toLocaleTimeString("en-US", { hour12: false });

    if (!prevStats.current) {
      // Initial load — generate boot sequence
      prevStats.current = stats;
      setEntries([
        { timestamp: now, level: "info", message: "INITIALIZING BOUNTY_HUNTER..." },
        { timestamp: now, level: "success", message: "CONNECTED TO PIPELINE GATEWAY." },
        {
          timestamp: now,
          level: "info",
          message: `DATABASE: ${stats.total} JOBS / ${stats.scored} SCORED / ${stats.relevant} RELEVANT / ${stats.tailored} TAILORED`,
        },
        { timestamp: now, level: "info", message: "SYSTEM STATUS: OPERATIONAL" },
      ]);
      return;
    }

    const prev = prevStats.current;
    const newEntries: typeof entries = [];

    if (stats.total > prev.total) {
      newEntries.push({
        timestamp: now,
        level: "success",
        message: `+${stats.total - prev.total} NEW JOBS SCRAPED (${stats.total} TOTAL)`,
      });
    }
    if (stats.scored > prev.scored) {
      newEntries.push({
        timestamp: now,
        level: "info",
        message: `SCORER: ${stats.scored - prev.scored} JOBS EVALUATED. ${stats.relevant} ABOVE THRESHOLD.`,
      });
    }
    if (stats.tailored > prev.tailored) {
      newEntries.push({
        timestamp: now,
        level: "success",
        message: `TAILOR: ${stats.tailored - prev.tailored} RESUMES GENERATED.`,
      });
    }

    if (newEntries.length > 0) {
      setEntries((prev) => [...prev.slice(-16), ...newEntries]);
    }

    prevStats.current = stats;
  }, [stats]);

  const addEntry = useCallback(
    (level: "info" | "warn" | "error" | "success", message: string) => {
      const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
      setEntries((prev) => [...prev.slice(-18), { timestamp, level, message }]);
    },
    []
  );

  return { entries, addEntry };
}
