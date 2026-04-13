"""Job scraping via python-jobspy."""

import sqlite3
from jobspy import scrape_jobs
from rich.console import Console
from rich.progress import Progress

from .db import insert_job

console = Console()


def run_scrape(
    conn: sqlite3.Connection,
    search_term: str,
    location: str | None,
    sites: list[str],
    results_wanted: int,
    country: str,
) -> int:
    """Scrape jobs from multiple boards and store in DB. Returns count of new jobs."""

    console.print(f"[bold blue]Scraping[/] '{search_term}' from {', '.join(sites)}...")

    try:
        df = scrape_jobs(
            site_name=sites,
            search_term=search_term,
            location=location or "",
            results_wanted=results_wanted,
            country_indeed=country,
            linkedin_fetch_description=True,
            verbose=0,
        )
    except Exception as e:
        console.print(f"[red]Scrape error:[/] {e}")
        return 0

    if df is None or df.empty:
        console.print("[yellow]No jobs found.[/]")
        return 0

    new_count = 0
    with Progress() as progress:
        task = progress.add_task("Storing jobs...", total=len(df))
        for _, row in df.iterrows():
            job_id = insert_job(
                conn,
                title=str(row.get("title", "")),
                company=str(row.get("company", "")) if row.get("company") else None,
                location=str(row.get("location", "")) if row.get("location") else None,
                url=str(row.get("job_url", "")) if row.get("job_url") else None,
                apply_url=str(row.get("job_url_direct", "")) if row.get("job_url_direct") else None,
                description=str(row.get("description", "")) if row.get("description") else None,
                salary=_format_salary(row),
                source=str(row.get("site", "")) if row.get("site") else None,
            )
            if job_id is not None:
                new_count += 1
            progress.advance(task)

    return new_count


def _format_salary(row) -> str | None:
    """Extract salary info from JobSpy row."""
    parts = []
    lo = row.get("min_amount")
    hi = row.get("max_amount")
    currency = row.get("currency", "USD")
    interval = row.get("interval")

    if lo and str(lo) != "nan":
        parts.append(f"{currency} {lo}")
    if hi and str(hi) != "nan":
        if parts:
            parts.append(f"- {hi}")
        else:
            parts.append(f"{currency} {hi}")
    if interval and str(interval) != "nan":
        parts.append(f"/{interval}")

    return " ".join(parts) if parts else None
