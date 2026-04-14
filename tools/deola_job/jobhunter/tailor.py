"""Two-phase LLM resume tailoring via Claude."""

import sqlite3
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from .config import RESUMES_DIR
from .db import Job, get_untailored_jobs, update_resume
from .llm import complete
from .resume import load_resume

console = Console()

ANALYZE_SYSTEM = "You are an expert recruiter analyzing job descriptions."

ANALYZE_PROMPT = """\
Analyze this job description and extract:
1. Top 5 required skills (exact terms from the posting)
2. Top 3 preferred/bonus skills
3. Key action verbs used in the description
4. Experience level expected
5. Industry/domain focus

Job: **{title}** at **{company}**

{description}

Respond in concise bullet points, no preamble."""

TAILOR_SYSTEM = """\
You are an expert resume writer. Follow these rules strictly:
- Do NOT invent experiences, skills, or metrics the candidate doesn't have
- Reword existing bullet points to mirror the job's language and keywords
- Reorder sections to front-load the most relevant experience
- Use "Action Verb + Metric + Outcome" format for bullet points where possible
- Incorporate the job's key terms naturally (ATS optimization)
- Keep the same section structure (Education, Experience, Skills, etc.)
- Output clean Markdown suitable for a professional resume"""

TAILOR_PROMPT = """\
Rewrite this resume to maximize fit for the target job.

## Job Analysis
{analysis}

## Original Resume
{resume}

Output ONLY the tailored resume in Markdown. No commentary, no explanations."""


def tailor_resumes(conn: sqlite3.Connection, model: str, resume_path: str | None,
                   min_score: int = 6) -> int:
    """Generate tailored resumes for all relevant, untailored jobs. Returns count."""
    resume_text = load_resume(resume_path)
    if not resume_text:
        console.print("[red]No resume found.[/] Set resume.path in config or pass --resume.")
        return 0

    jobs = get_untailored_jobs(conn, min_score)
    if not jobs:
        console.print("[yellow]No jobs to tailor for.[/] Run 'filter' first.")
        return 0

    try:
        from .llm import get_client
        get_client()
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        return 0

    console.print(f"[bold blue]Tailoring[/] resumes for {len(jobs)} jobs with {model}...")
    count = 0

    with Progress() as progress:
        task = progress.add_task("Tailoring...", total=len(jobs))
        for job in jobs:
            try:
                path = _tailor_one(model, resume_text, job)
                update_resume(conn, job.id, str(path))
                console.print(f"  [green]Done[/] {job.title} @ {job.company or '?'} → {path.name}")
                count += 1
            except Exception as e:
                console.print(f"  [red]Error tailoring {job.title}:[/] {e}")
            progress.advance(task)

    return count


def _tailor_one(model: str, resume_text: str, job: Job) -> Path:
    """Two-phase tailor: analyze JD, then rewrite resume. Returns path to saved file."""
    desc = (job.description or "No description available")[:3000]

    # Phase 1: Analyze the job description
    analysis = complete(
        model=model,
        system=ANALYZE_SYSTEM,
        user=ANALYZE_PROMPT.format(
            title=job.title,
            company=job.company or "Unknown",
            description=desc,
        ),
        max_tokens=500,
        temperature=0.2,
    )

    # Phase 2: Tailor the resume
    tailored = complete(
        model=model,
        system=TAILOR_SYSTEM,
        user=TAILOR_PROMPT.format(
            analysis=analysis,
            resume=resume_text[:3000],
        ),
        max_tokens=2000,
        temperature=0.3,
    )

    # Strip markdown fences if wrapped
    if tailored.startswith("```"):
        tailored = tailored.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Save to file
    safe_name = f"{job.id}_{_slugify(job.company)}_{_slugify(job.title)}.md"
    out_path = RESUMES_DIR / safe_name
    out_path.write_text(tailored, encoding="utf-8")
    return out_path


def _slugify(text: str | None) -> str:
    if not text:
        return "unknown"
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:30].strip("_")
