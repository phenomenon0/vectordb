"""LLM-based job relevance scoring via Claude."""

import json
import sqlite3
from rich.console import Console
from rich.progress import Progress

from .db import Job, get_unscored_jobs, update_score
from .llm import complete
from .resume import load_resume

console = Console()

SYSTEM_PROMPT = """\
You are an expert recruiter evaluating job fit. You will receive a candidate's resume \
and a job description. Score the fit from 1 to 10 and explain why.

Scoring guide:
- 9-10: Perfect match — skills, experience level, and domain all align
- 7-8: Strong match — most requirements met, minor gaps
- 5-6: Partial match — some relevant skills but significant gaps
- 3-4: Weak match — few overlapping skills
- 1-2: No match — completely different field or level

Respond with ONLY valid JSON, no markdown fences:
{"score": <int 1-10>, "reason": "<1-2 sentence explanation>"}"""

USER_PROMPT = """\
## Candidate Resume
{resume}

## Job Description
**{title}** at **{company}**
Location: {location}

{description}"""


def score_jobs(conn: sqlite3.Connection, model: str, resume_path: str | None) -> int:
    """Score all unscored jobs against the resume. Returns count scored."""
    resume_text = load_resume(resume_path)
    if not resume_text:
        console.print("[red]No resume found.[/] Set resume.path in config or pass --resume.")
        return 0

    jobs = get_unscored_jobs(conn)
    if not jobs:
        console.print("[yellow]No unscored jobs. Run 'scrape' first.[/]")
        return 0

    # Validate API key early
    try:
        from .llm import get_client
        get_client()
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        return 0

    console.print(f"[bold blue]Scoring[/] {len(jobs)} jobs with {model}...")
    scored = 0

    with Progress() as progress:
        task = progress.add_task("Scoring...", total=len(jobs))
        for job in jobs:
            try:
                result = _score_one(model, resume_text, job)
                update_score(conn, job.id, result["score"], result["reason"])
                label = "[green]" if result["score"] >= 6 else "[dim]"
                console.print(f"  {label}{result['score']:2d}[/] {job.title} @ {job.company or '?'} — {result['reason']}")
                scored += 1
            except Exception as e:
                console.print(f"  [red]Error scoring {job.title}:[/] {e}")
            progress.advance(task)

    return scored


def _score_one(model: str, resume_text: str, job: Job) -> dict:
    """Score a single job. Returns {"score": int, "reason": str}."""
    user_msg = USER_PROMPT.format(
        resume=resume_text[:3000],
        title=job.title,
        company=job.company or "Unknown",
        location=job.location or "Not specified",
        description=(job.description or "No description")[:2000],
    )

    text = complete(
        model=model,
        system=SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=200,
        temperature=0.1,
    )

    # Strip markdown fences if the model adds them
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)
