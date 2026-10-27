from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

PAPER_DIR = Path(__file__).resolve().parent
SOURCE = PAPER_DIR / "paper.md"
OUTPUT = PAPER_DIR / "paper.pdf"
CHROME_CANDIDATES = (
    os.environ.get("CHROME_BIN"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    raise SystemExit("Set CHROME_BIN to a Chromium-based browser executable")


def main() -> None:
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
@page {{ size: Letter; margin: 0.85in 0.82in 0.85in; }}
html, body {{ background: #fff !important; color: #202626 !important; color-scheme: light; }}
body {{ font: 11pt/1.5 Georgia, "Times New Roman", serif; }}
h1 {{ margin: 0 0 9pt; font: 700 22pt/1.12 Georgia, serif; letter-spacing: -.3pt; }}
h2 {{ margin: 19pt 0 7pt; font: 700 14pt/1.2 Arial, sans-serif; break-after: avoid; }}
h3 {{ margin: 14pt 0 5pt; font: 700 11.5pt/1.2 Arial, sans-serif; break-after: avoid; }}
p {{ margin: 0 0 9pt; text-align: justify; orphans: 3; widows: 3; }}
ul, ol {{ margin: 4pt 0 8pt 17pt; padding: 0; }}
li {{ margin: 2pt 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 7pt 0 4pt; font: 7.3pt/1.2 Arial, sans-serif; }}
th, td {{ border: .5pt solid #697574; padding: 4pt 3pt; text-align: right; }}
th {{ background: #eaf0ed; font-weight: 700; }}
th:first-child, td:first-child, th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
tr {{ break-inside: avoid; }}
a {{ color: #155d4d; text-decoration: none; }}
code {{ font: 8pt Menlo, monospace; }}
.bn {{ font-family: "Kohinoor Bangla", "Bangla Sangam MN", sans-serif; }}
strong {{ font-weight: 700; }}
hr {{ border: 0; border-top: .6pt solid #aab4b0; margin: 8pt 0; }}
</style></head><body>{body}</body></html>"""
    with tempfile.TemporaryDirectory(prefix="bangla-paper-") as temp_dir:
        html_path = Path(temp_dir) / "paper.html"
        html_path.write_text(document, encoding="utf-8")
        subprocess.run(
            [
                find_chrome(),
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--allow-file-access-from-files",
                f"--print-to-pdf={OUTPUT}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
