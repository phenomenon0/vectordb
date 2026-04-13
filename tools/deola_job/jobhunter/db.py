"""SQLite database for job tracking."""

import sqlite3
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .config import DB_PATH, ensure_dirs

SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    company     TEXT,
    location    TEXT,
    url         TEXT UNIQUE,
    apply_url   TEXT,
    description TEXT,
    salary      TEXT,
    source      TEXT,
    scraped_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id),
    status      TEXT NOT NULL DEFAULT 'scraped',
    fit_score   INTEGER,
    fit_reason  TEXT,
    resume_path TEXT,
    scored_at   TEXT,
    tailored_at TEXT,
    applied_at  TEXT,
    UNIQUE(job_id)
);
"""


@dataclass
class Job:
    id: int
    title: str
    company: str | None
    location: str | None
    url: str | None
    apply_url: str | None
    description: str | None
    salary: str | None
    source: str | None
    scraped_at: str


@dataclass
class Application:
    id: int
    job_id: int
    status: str
    fit_score: int | None
    fit_reason: str | None
    resume_path: str | None
    scored_at: str | None
    tailored_at: str | None
    applied_at: str | None


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def insert_job(conn: sqlite3.Connection, *, title: str, company: str | None,
               location: str | None, url: str | None, apply_url: str | None,
               description: str | None, salary: str | None,
               source: str | None) -> int | None:
    """Insert a job, skip if URL already exists. Returns job id or None if duplicate."""
    now = datetime.now().isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO jobs (title, company, location, url, apply_url, description, salary, source, scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (title, company, location, url, apply_url, description, salary, source, now),
        )
        job_id = cur.lastrowid
        conn.execute(
            "INSERT INTO applications (job_id, status) VALUES (?, 'scraped')",
            (job_id,),
        )
        conn.commit()
        return job_id
    except sqlite3.IntegrityError:
        return None


def get_unscored_jobs(conn: sqlite3.Connection) -> list[Job]:
    """Get jobs that haven't been scored yet."""
    rows = conn.execute(
        "SELECT j.* FROM jobs j "
        "JOIN applications a ON a.job_id = j.id "
        "WHERE a.fit_score IS NULL "
        "ORDER BY j.scraped_at DESC"
    ).fetchall()
    return [Job(**dict(r)) for r in rows]


def update_score(conn: sqlite3.Connection, job_id: int, score: int, reason: str):
    """Set fit score and reason for a job."""
    now = datetime.now().isoformat()
    status = "relevant" if score >= 6 else "filtered"
    conn.execute(
        "UPDATE applications SET fit_score=?, fit_reason=?, scored_at=?, status=? WHERE job_id=?",
        (score, reason, now, status, job_id),
    )
    conn.commit()


def get_untailored_jobs(conn: sqlite3.Connection, min_score: int = 6) -> list[Job]:
    """Get relevant jobs that don't have a tailored resume yet."""
    rows = conn.execute(
        "SELECT j.* FROM jobs j "
        "JOIN applications a ON a.job_id = j.id "
        "WHERE a.fit_score >= ? AND a.resume_path IS NULL "
        "ORDER BY a.fit_score DESC",
        (min_score,),
    ).fetchall()
    return [Job(**dict(r)) for r in rows]


def update_resume(conn: sqlite3.Connection, job_id: int, resume_path: str):
    """Set the tailored resume path for a job."""
    now = datetime.now().isoformat()
    conn.execute(
        "UPDATE applications SET resume_path=?, tailored_at=?, status='tailored' WHERE job_id=?",
        (resume_path, now, job_id),
    )
    conn.commit()


def get_all_applications(conn: sqlite3.Connection) -> list[dict]:
    """Get all jobs with their application status for the dashboard."""
    rows = conn.execute(
        "SELECT j.id, j.title, j.company, j.location, j.url, j.apply_url, j.salary, j.source, "
        "a.status, a.fit_score, a.fit_reason, a.resume_path, a.scored_at, a.tailored_at "
        "FROM jobs j JOIN applications a ON a.job_id = j.id "
        "ORDER BY a.fit_score DESC NULLS LAST, j.scraped_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get summary statistics."""
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    scored = conn.execute("SELECT COUNT(*) FROM applications WHERE fit_score IS NOT NULL").fetchone()[0]
    relevant = conn.execute("SELECT COUNT(*) FROM applications WHERE status IN ('relevant', 'tailored')").fetchone()[0]
    tailored = conn.execute("SELECT COUNT(*) FROM applications WHERE resume_path IS NOT NULL").fetchone()[0]
    return {"total": total, "scored": scored, "relevant": relevant, "tailored": tailored}
