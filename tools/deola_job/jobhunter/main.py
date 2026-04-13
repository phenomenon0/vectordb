"""JobHunter CLI — scrape, filter, tailor, track."""

from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, init_config, CONFIG_FILE
from .db import get_conn, get_all_applications, get_stats

app = typer.Typer(
    name="jobhunter",
    help="AI-powered job hunting: scrape → filter → tailor → track",
    no_args_is_help=True,
)
console = Console()


@app.command()
def scrape(
    role: str = typer.Argument(..., help="Job title to search for"),
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Location filter"),
    sites: Optional[str] = typer.Option(None, "--sites", "-s", help="Comma-separated sites: linkedin,indeed,glassdoor,zip_recruiter,google"),
    count: Optional[int] = typer.Option(None, "--count", "-n", help="Number of results per site"),
    country: Optional[str] = typer.Option(None, "--country", help="Country code for Indeed"),
):
    """Scrape job listings from multiple boards."""
    from .scraper import run_scrape

    cfg = load_config()
    conn = get_conn()

    site_list = sites.split(",") if sites else cfg.scrape.sites
    n = count or cfg.scrape.results_wanted
    c = country or cfg.scrape.country

    new = run_scrape(conn, search_term=role, location=location, sites=site_list,
                     results_wanted=n, country=c)

    stats = get_stats(conn)
    console.print(f"\n[green]{new} new jobs[/] added ({stats['total']} total in DB)")
    conn.close()


@app.command()
def filter(
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Path to resume file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
):
    """Score jobs for relevance against your resume."""
    from .filter import score_jobs

    cfg = load_config()
    conn = get_conn()

    r = resume or cfg.resume.path
    m = model or cfg.llm.model

    scored = score_jobs(conn, model=m, resume_path=r)

    stats = get_stats(conn)
    console.print(f"\n[green]{scored} jobs scored[/] — {stats['relevant']} relevant, "
                  f"{stats['total'] - stats['relevant']} filtered out")
    conn.close()


@app.command()
def tailor(
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Path to resume file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    min_score: Optional[int] = typer.Option(None, "--min-score", help="Minimum fit score to tailor for"),
):
    """Generate tailored resumes for relevant jobs."""
    from .tailor import tailor_resumes

    cfg = load_config()
    conn = get_conn()

    r = resume or cfg.resume.path
    m = model or cfg.llm.model
    ms = min_score if min_score is not None else cfg.llm.min_score

    count = tailor_resumes(conn, model=m, resume_path=r, min_score=ms)

    console.print(f"\n[green]{count} resumes tailored[/]")
    conn.close()


@app.command()
def status(
    top: int = typer.Option(20, "--top", "-n", help="Number of jobs to show"),
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Show all jobs including filtered"),
):
    """Show job tracking dashboard."""
    conn = get_conn()
    stats = get_stats(conn)

    # Summary bar
    console.print()
    console.print(f"[bold]Jobs:[/] {stats['total']}  "
                  f"[bold]Scored:[/] {stats['scored']}  "
                  f"[bold green]Relevant:[/] {stats['relevant']}  "
                  f"[bold cyan]Tailored:[/] {stats['tailored']}")
    console.print()

    apps = get_all_applications(conn)
    if not apps:
        console.print("[yellow]No jobs yet. Run 'jobhunter scrape' first.[/]")
        conn.close()
        return

    if not all_jobs:
        apps = [a for a in apps if a["status"] != "filtered"]

    apps = apps[:top]

    table = Table(show_lines=False, pad_edge=False)
    table.add_column("Score", style="bold", width=5, justify="right")
    table.add_column("Status", width=8)
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Company", max_width=20)
    table.add_column("Location", max_width=20)
    table.add_column("Salary", max_width=15)
    table.add_column("Resume", max_width=8)

    for a in apps:
        score_str = str(a["fit_score"]) if a["fit_score"] is not None else "-"
        status_style = {
            "scraped": "dim",
            "filtered": "red",
            "relevant": "green",
            "tailored": "bold green",
        }.get(a["status"], "white")

        resume_flag = "[green]yes[/]" if a["resume_path"] else ""

        table.add_row(
            score_str,
            f"[{status_style}]{a['status']}[/]",
            a["title"] or "",
            a["company"] or "",
            a["location"] or "",
            a["salary"] or "",
            resume_flag,
        )

    console.print(table)
    conn.close()


@app.command()
def init():
    """Create default config file."""
    init_config()
    console.print(f"[green]Config created at[/] {CONFIG_FILE}")
    console.print("Edit it to set your resume path and LLM model.")


@app.command()
def show(
    job_id: int = typer.Argument(..., help="Job ID to show details for"),
):
    """Show full details for a specific job."""
    conn = get_conn()
    row = conn.execute(
        "SELECT j.*, a.status, a.fit_score, a.fit_reason, a.resume_path "
        "FROM jobs j JOIN applications a ON a.job_id = j.id WHERE j.id = ?",
        (job_id,),
    ).fetchone()

    if not row:
        console.print(f"[red]Job {job_id} not found.[/]")
        conn.close()
        return

    r = dict(row)
    console.print(f"\n[bold cyan]{r['title']}[/] @ [bold]{r['company'] or '?'}[/]")
    console.print(f"Location: {r['location'] or 'N/A'}  |  Salary: {r['salary'] or 'N/A'}")
    console.print(f"Source: {r['source']}  |  Status: {r['status']}")

    if r["fit_score"] is not None:
        console.print(f"Fit Score: [bold]{r['fit_score']}/10[/] — {r['fit_reason']}")

    if r["url"]:
        console.print(f"Job URL: {r['url']}")
    if r["apply_url"]:
        console.print(f"Apply: {r['apply_url']}")
    if r["resume_path"]:
        console.print(f"Resume: {r['resume_path']}")

    if r.get("description"):
        console.print(f"\n[dim]{'─' * 60}[/]")
        console.print(r["description"][:1000])
        if len(r["description"]) > 1000:
            console.print("[dim]... (truncated)[/]")

    conn.close()


if __name__ == "__main__":
    app()
