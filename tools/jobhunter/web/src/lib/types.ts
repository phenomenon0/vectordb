export type PipelineStatus = "scraped" | "filtered" | "relevant" | "tailored";
export type TrackingStatus = "applied" | "interviewing" | "offer" | "rejected" | "withdrawn";
export type JobStatus = PipelineStatus | TrackingStatus;

export const VALID_STATUSES: JobStatus[] = [
  "scraped", "filtered", "relevant", "tailored",
  "applied", "interviewing", "offer", "rejected", "withdrawn",
];

export const KANBAN_COLUMNS = [
  { key: "applied" as const, label: "Applied" },
  { key: "interviewing" as const, label: "Interviewing" },
  { key: "offer" as const, label: "Offer" },
  { key: "rejected" as const, label: "Rejected" },
] as const;

export interface JobRow {
  id: number;
  title: string;
  company: string | null;
  location: string | null;
  url: string | null;
  apply_url: string | null;
  description: string | null;
  salary: string | null;
  source: string | null;
  scraped_at: string;
  status: JobStatus;
  fit_score: number | null;
  fit_reason: string | null;
  resume_path: string | null;
}

export interface JobDetail extends JobRow {
  resumeContent: string | null;
}

export interface Stats {
  total: number;
  scored: number;
  relevant: number;
  tailored: number;
}
