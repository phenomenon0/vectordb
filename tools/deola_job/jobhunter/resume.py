"""Resume loading — supports PDF, TXT, and Markdown."""

from pathlib import Path
from rich.console import Console

console = Console()


def load_resume(path: str | None) -> str | None:
    """Load resume text from a file path. Returns None if not found."""
    if not path:
        return None

    p = Path(path).expanduser()
    if not p.exists():
        console.print(f"[red]Resume not found:[/] {p}")
        return None

    if p.suffix.lower() == ".pdf":
        return _load_pdf(p)
    else:
        return p.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str | None:
    """Extract text from a PDF resume."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts) if text_parts else None
    except ImportError:
        console.print("[red]pdfplumber not installed.[/] Run: pip install pdfplumber")
        return None
    except Exception as e:
        console.print(f"[red]Error reading PDF:[/] {e}")
        return None
