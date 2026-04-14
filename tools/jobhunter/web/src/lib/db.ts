import Database from "better-sqlite3";
import { getUserDbPath } from "./user-data";

// Cache DB connections per user
const _dbs = new Map<string, Database.Database>();

export function getDb(userId: string): Database.Database {
  let db = _dbs.get(userId);
  if (!db) {
    const dbPath = getUserDbPath(userId);
    db = new Database(dbPath, { readonly: false });
    db.pragma("journal_mode = WAL");

    // Auto-create tables if new user
    db.exec(`
      CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        company TEXT,
        location TEXT,
        url TEXT,
        apply_url TEXT,
        description TEXT,
        salary TEXT,
        source TEXT,
        scraped_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
        status TEXT NOT NULL DEFAULT 'scraped',
        fit_score INTEGER,
        fit_reason TEXT,
        resume_path TEXT,
        scored_at TEXT,
        tailored_at TEXT,
        applied_at TEXT
      );
    `);

    _dbs.set(userId, db);
  }
  return db;
}

export interface Job {
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
}

export interface Application {
  id: number;
  job_id: number;
  status: string;
  fit_score: number | null;
  fit_reason: string | null;
  resume_path: string | null;
  scored_at: string | null;
  tailored_at: string | null;
  applied_at: string | null;
}

export interface JobWithApplication extends Job {
  status: string;
  fit_score: number | null;
  fit_reason: string | null;
  resume_path: string | null;
}

export function getAllJobs(userId: string): JobWithApplication[] {
  const db = getDb(userId);
  return db
    .prepare(
      `SELECT j.id, j.title, j.company, j.location, j.url, j.apply_url,
              j.salary, j.source, j.scraped_at, j.description,
              a.status, a.fit_score, a.fit_reason, a.resume_path
       FROM jobs j JOIN applications a ON a.job_id = j.id
       ORDER BY a.fit_score DESC, j.scraped_at DESC`
    )
    .all() as JobWithApplication[];
}

export function getJob(userId: string, id: number): (JobWithApplication & { description: string | null }) | null {
  const db = getDb(userId);
  return (
    db
      .prepare(
        `SELECT j.*, a.status, a.fit_score, a.fit_reason, a.resume_path
         FROM jobs j JOIN applications a ON a.job_id = j.id
         WHERE j.id = ?`
      )
      .get(id) as (JobWithApplication & { description: string | null }) | null
  );
}

export function updateJobStatus(userId: string, jobId: number, status: string): void {
  const db = getDb(userId);
  const result = db
    .prepare(
      `UPDATE applications
       SET status = ?,
           applied_at = CASE WHEN ? = 'applied' AND applied_at IS NULL THEN datetime('now') ELSE applied_at END
       WHERE job_id = ?`
    )
    .run(status, status, jobId);
  if (result.changes === 0) {
    throw new Error(`Job ${jobId} not found`);
  }
}

export function getStats(userId: string) {
  const db = getDb(userId);
  const total = (db.prepare("SELECT COUNT(*) as c FROM jobs").get() as { c: number }).c;
  const scored = (
    db.prepare("SELECT COUNT(*) as c FROM applications WHERE fit_score IS NOT NULL").get() as { c: number }
  ).c;
  const relevant = (
    db
      .prepare("SELECT COUNT(*) as c FROM applications WHERE status IN ('relevant', 'tailored')")
      .get() as { c: number }
  ).c;
  const tailored = (
    db.prepare("SELECT COUNT(*) as c FROM applications WHERE resume_path IS NOT NULL").get() as { c: number }
  ).c;
  return { total, scored, relevant, tailored };
}
