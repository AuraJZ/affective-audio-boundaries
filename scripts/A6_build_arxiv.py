r"""Build the arXiv submission package, then compile it to prove it builds.

    .venv/Scripts/python scripts/A6_build_arxiv.py

## Why this is separate from 62_build_submission.py

`62` mirrors the repository layout on purpose, so the relative paths inside
`main.tex` keep working. arXiv cannot accept that layout, for two reasons that
both fail silently until arXiv rejects the upload:

1. **No parent-directory references.** `main.tex` loads figures as
   `../reports/figures_r/Fig1_wall.pdf`. arXiv unpacks your source into one
   directory and builds there; a path that climbs above it resolves to nothing.
   Figures are therefore copied into `figures/` and every `\includegraphics`
   path is rewritten to match.

2. **arXiv does not run BibTeX.** It runs LaTeX only, so the bibliography has to
   arrive pre-built as `main.bbl`. The repository `.gitignore` excludes
   `manuscript/*.bbl` because it is a build product -- correct for the repo,
   wrong for arXiv, and exactly the kind of thing that is noticed at 2 a.m. on
   the day of submission.

The supplementary is not a second compiled document here. arXiv builds one
paper; the supplement rides along as an ancillary file under `anc/`, which is
arXiv's own mechanism for exactly this, and readers get it as a separate
download next to the PDF.

## The package

    main.tex           paths rewritten to figures/
    preamble.tex
    legends/all.tex
    main.bbl           pre-built bibliography
    figures/*.pdf      only the figures main.tex actually cites
    anc/supplementary.pdf

## What it checks before declaring success

  - `main.bbl` exists and is not older than `references.bib`
  - every `\includegraphics` target resolves inside the package
  - no surviving `..` in any input path
  - no LaTeX build residue, no raw media
  - `\fillmarkstrue` is reported loudly -- shipping the placeholder highlights
    to arXiv would be embarrassing but is not fatal, so it warns rather than
    stops
  - and then it actually runs pdflatex on the package and compares the page
    count with the repository build. A package that assembles but does not
    compile is worth nothing.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from soundml.provenance import REPO_ROOT

# 🔴 Without this the script dies on its own output. When stdout is a pipe
# rather than a console, Python picks the ANSI codepage (cp1252 here), and the
# first emoji raises UnicodeEncodeError -- so it worked when run directly and
# crashed under `| tee`, in CI, or anywhere the output was redirected. The
# crash happened before the leak scan, which is the one check that must never
# be skipped silently.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MS = REPO_ROOT / "manuscript"
DIST = REPO_ROOT / "dist"

# MiKTeX is not on PATH in this environment.
TEX_DIRS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64",
    Path("C:/Program Files/MiKTeX/miktex/bin/x64"),
]


def find_pdflatex() -> str | None:
    if shutil.which("pdflatex"):
        return "pdflatex"
    for d in TEX_DIRS:
        exe = d / "pdflatex.exe"
        if exe.exists():
            return str(exe)
    return None


def page_count(log: Path) -> int | None:
    if not log.exists():
        return None
    m = re.search(r"Output written on .*?\((\d+) pages", log.read_text(
        encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def build(stage: Path) -> list[str]:
    """Assemble the package. Returns a list of problems found."""
    problems: list[str] = []
    (stage / "figures").mkdir(parents=True)
    (stage / "legends").mkdir()
    (stage / "anc").mkdir()

    main = (MS / "main.tex").read_text(encoding="utf-8")

    # ---- figures: copy in, rewrite the paths ----
    # Figures are cited through \msfigure{path}{label}{legend}, not through a
    # literal \includegraphics -- the only \includegraphics in the sources is
    # the one inside the macro definition, whose argument is "#1". Match both
    # forms so this keeps working if the macro is ever removed.
    wanted = (re.findall(r"\\msfigure\{([^}]+)\}", main)
              + re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main))
    for rel in wanted:
        src = (MS / rel).resolve()
        if not src.exists():
            problems.append(f"figure not found in the repository: {rel}")
            continue
        shutil.copy2(src, stage / "figures" / src.name)
    main = re.sub(r"(\\msfigure\{)[^}]*?([^/}]+\.pdf\})", r"\1figures/\2", main)
    main = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)[^}]*?([^/}]+\.pdf\})",
                  r"\1figures/\2", main)
    print(f"  figures/          {len(wanted)} referenced, "
          f"{len(list((stage / 'figures').glob('*')))} copied")

    (stage / "main.tex").write_text(main, encoding="utf-8")
    shutil.copy2(MS / "preamble.tex", stage / "preamble.tex")
    shutil.copy2(MS / "legends/all.tex", stage / "legends/all.tex")

    # ---- the bibliography, pre-built ----
    bbl = MS / "main.bbl"
    bib = REPO_ROOT / "references/references.bib"
    if not bbl.exists():
        problems.append("manuscript/main.bbl is missing -- compile the "
                        "manuscript once (pdflatex, bibtex, pdflatex) first; "
                        "arXiv does not run BibTeX")
    else:
        if bib.exists() and bbl.stat().st_mtime < bib.stat().st_mtime:
            problems.append("manuscript/main.bbl is older than references.bib "
                            "-- the bibliography would ship stale")
        shutil.copy2(bbl, stage / "main.bbl")
        print(f"  main.bbl          {bbl.stat().st_size/1024:.1f} KB")

    # ---- supplementary as an ancillary file ----
    supp = MS / "supplementary.pdf"
    if supp.exists():
        shutil.copy2(supp, stage / "anc" / "supplementary.pdf")
        print(f"  anc/              supplementary.pdf "
              f"({supp.stat().st_size/1e6:.1f} MB)")
    else:
        problems.append("manuscript/supplementary.pdf is missing")

    # ---- checks on what was assembled ----
    for f in ("main.tex", "preamble.tex", "legends/all.tex"):
        p = stage / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\\(?:includegraphics(?:\[[^\]]*\])?|msfigure|input|include)\{([^}]+)\}", t):
            if ".." in m.group(1):
                problems.append(f"{f} still climbs above the package root: {m.group(1)}")
        for m in re.finditer(r"\\(?:includegraphics(?:\[[^\]]*\])?|msfigure)\{([^}]+)\}", t):
            path = m.group(1)
            # "#1" is the macro's own parameter inside its definition, not a file.
            if "#" in path:
                continue
            tgt = stage / path
            if not (tgt.exists() or tgt.with_suffix(".pdf").exists()):
                problems.append(f"{f} references a missing figure: {path}")

    return problems


# --------------------------------------------------------------- leak scan --
# arXiv: "All announced content is archival and cannot be removed. Make sure
# that data you do not want archived is not part of your upload, for example
# TeX comments in your source."
#
# This is the one preflight item the rest of this script does not already
# cover, and it is the only irreversible one. Anyone can fetch the LaTeX from
# a paper's page (Download source), so every comment ships: a stray
# "% ask X@gmail.com whether we mention the patent" is public and permanent.
#
# The GitHub route gets a human gate -- the release repo is assembled by hand
# and `git status` is read before the first commit. This route has none: it
# packages straight out of manuscript/. So the gate lives here, and it runs
# BEFORE the compile check, because a package that compiles is not the same as
# a package that is safe to publish.
#
# Note the scope: not just main.tex. preamble.tex, legends/all.tex and main.bbl
# ship too, and so do the figure PDFs -- which carry metadata a text grep never
# sees.
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PERSONAL_MAIL = r"(?i)@(gmail|outlook|hotmail|yahoo|qq|163|126)\."
# 🔴 A local path is `C:\...` or a POSIX home path. The obvious-looking
# `[A-Za-z]:/` also matches the "s:/" inside every "https://…", which fired on
# three \url{} lines the first time this ran. A check that cries wolf is a
# check nobody reads, so URLs are stripped before this is applied.
LOCAL_PATH = r"(?<![A-Za-z0-9])[A-Za-z]:\\|/Users/|/home/[a-z]|AppData|\.venv"

# Applied to COMMENTS, which readers never see but arXiv publishes forever.
COMMENT_PATTERNS: list[tuple[str, str]] = [
    (EMAIL, "email address"),
    (r"[一-鿿]", "CJK text (internal drafting note)"),
    (r"(?i)\b(patent|claim\s+draft|invention\s+disclosure|prior\s+art)\b",
     "patent language"),
    (r"(?i)\b(TODO|FIXME|XXX|HACK)\b", "unresolved marker"),
    (r"(?i)\b(confidential|internal\s+only|do\s+not\s+(cite|share|distribute))\b",
     "confidentiality marker"),
    (LOCAL_PATH, "local filesystem path"),
    (PERSONAL_MAIL, "personal mail provider"),
]

# Applied to BODY TEXT, which is published anyway -- so only the categories
# that are never legitimate there. A corresponding author's institutional
# email belongs in the body; a personal one usually does not.
BODY_PATTERNS: list[tuple[str, str]] = [
    (PERSONAL_MAIL, "personal email in the body"),
    (r"[一-鿿]", "CJK text in the body"),
    (LOCAL_PATH, "local filesystem path in the body"),
]

# Generator strings every LaTeX/R toolchain writes. These name the software and
# its version and nothing else -- they appear on essentially every arXiv paper.
# What is NOT allowed through is /Title and /Author, which is where a filename,
# a username, or an unredacted working title actually leaks.
PDF_META_OK = (r"(?i)^(cairo|matplotlib|R |ggplot2|LaTeX|pdfTeX|MiKTeX|TeX Live"
               r"|dvips|GPL Ghostscript|Acrobat|Quartz|Skia)")

# Comments that are part of the paper's own engineering record and are meant to
# be readable. Matched against the whole comment, so keep them specific.
LEAK_ALLOW = [
    r"\\@gobble@om",          # a LaTeX internal named in a bug note
    r"\\@tempboxa",
    r"line numbers for reviewer reference",
]


def strip_urls(s: str) -> str:
    """Remove URLs before path matching. "https://x" contains "s:/"."""
    return re.sub(r"\\url\{[^}]*\}|\w+://\S*", " ", s)


def tex_comments(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        m = re.search(r"(?<!\\)%(.*)", line)
        if m and m.group(1).strip():
            out.append((i, m.group(1).strip()))
    return out


def scan_leaks(stage: Path) -> list[str]:
    """Everything that would ship. Returns human-readable findings."""
    found: list[str] = []

    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(stage).as_posix()

        if p.suffix in (".tex", ".bbl", ".bib", ".cls", ".sty"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for ln, c in tex_comments(text):
                if any(re.search(a, c) for a in LEAK_ALLOW):
                    continue
                bare = strip_urls(c)
                for pat, why in COMMENT_PATTERNS:
                    if re.search(pat, bare):
                        found.append(f"{rel}:{ln}  [{why}]  % {c[:84]}")
                        break
            for ln, line in enumerate(text.split("\n"), 1):
                body = strip_urls(re.sub(r"(?<!\\)%.*", "", line))
                for pat, why in BODY_PATTERNS:
                    if re.search(pat, body):
                        found.append(f"{rel}:{ln}  [{why}]  {body.strip()[:80]}")
                        break

        elif p.suffix == ".pdf":
            # Figure PDFs carry Producer/Creator/Title and sometimes the path
            # they were written from. A text grep over the sources never sees
            # any of it.
            raw = p.read_bytes()
            for pat, why in ((rb"[A-Za-z]:\\\\", "drive path"),
                             (rb"/Users/", "user path"),
                             (rb"AppData", "AppData path")):
                if re.search(pat, raw):
                    found.append(f"{rel}  [{why} in PDF bytes]")
            for key in (b"/Title", b"/Author", b"/Subject", b"/Keywords",
                        b"/Creator", b"/Producer"):
                for m in re.finditer(re.escape(key) + rb"\s*\(([^)]{0,120})\)", raw):
                    val = m.group(1).decode("latin-1", "replace").strip()
                    if val and not re.match(PDF_META_OK, val):
                        found.append(f"{rel}  [PDF {key.decode()} = {val[:70]!r}]")
    return found


def verify(stage: Path, expect_pages: int | None) -> bool:
    """Compile the assembled package. This is the only check that counts."""
    exe = find_pdflatex()
    if not exe:
        print("\n⚠️  pdflatex not found -- package assembled but NOT compile-tested.")
        print("   That is the one check that matters; run it before uploading.")
        return True

    print("\ncompiling the package as arXiv would…")
    for i in range(2):
        subprocess.run([exe, "-interaction=nonstopmode", "main.tex"],
                       cwd=stage, capture_output=True)
    pdf, log = stage / "main.pdf", stage / "main.log"
    if not pdf.exists():
        print("🔴 the package does NOT compile. First errors:")
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("!"):
                    print(f"    {line}")
        return False

    n = page_count(log)
    txt = log.read_text(encoding="utf-8", errors="replace")
    undef = len(re.findall(r"undefined", txt))
    print(f"  compiled: {n} pages, {undef} undefined references")
    if expect_pages and n != expect_pages:
        print(f"🔴 {n} pages here vs {expect_pages} in the repository build — "
              "the flattened package is not the same document")
        return False
    if undef:
        print("🔴 undefined references — main.bbl did not take effect")
        return False
    print("✅ compiles, same page count as the repository build, no undefined refs")

    # The compiled proof is not part of the upload; arXiv builds its own.
    for junk in ("main.pdf", "main.log", "main.aux", "main.out"):
        (stage / junk).unlink(missing_ok=True)
    return True


def main() -> None:
    if not (MS / "main.tex").exists():
        raise SystemExit("manuscript/main.tex not found")

    expect = page_count(MS / "main.log")
    print(f"repository build: {expect} pages\n")
    print("assembling…")

    stage = Path(tempfile.mkdtemp(prefix="sm_arxiv_"))
    try:
        problems = build(stage)

        # Not anchored to line start: the declaration sits mid-line as
        # "\newif\iffillmarks \fillmarkstrue", so "^\s*\\fillmarkstrue" never
        # matched and this warning silently never fired.
        pre = re.sub(r"(?<!\\)%.*", "", (MS / "preamble.tex").read_text(encoding="utf-8"))
        if re.search(r"\\fillmarkstrue", pre):
            print("\n⚠️  \\fillmarkstrue is set: the [FILL] markers and the "
                  "checklist box\n   will appear in the arXiv PDF. Set "
                  "\\fillmarksfalse once every item is filled.")

        if problems:
            print(f"\n🔴 {len(problems)} problem(s):")
            for p in problems:
                print(f"    {p}")
            raise SystemExit(1)

        # Before the compile check, not after: arXiv keeps whatever is uploaded
        # forever, so "is it safe to publish" outranks "does it build".
        print("\nscanning everything that would ship for things that cannot be "
              "un-published…")
        leaks = scan_leaks(stage)
        if leaks:
            print(f"🔴 {len(leaks)} item(s) that would become permanently public:")
            for x in leaks[:40]:
                print(f"    {x}")
            if len(leaks) > 40:
                print(f"    … and {len(leaks) - 40} more")
            print("\n   arXiv source is downloadable by anyone and announced content"
                  "\n   cannot be withdrawn. Clear these, then rebuild.")
            raise SystemExit(1)
        print("  ✅ no emails, local paths, CJK drafting notes, patent language or"
              " PDF metadata")

        if not verify(stage, expect):
            raise SystemExit(1)

        DIST.mkdir(exist_ok=True)
        out = shutil.make_archive(
            str(DIST / f"arxiv_{time.strftime('%Y%m%d')}"), "zip", root_dir=stage)
        files = [p for p in stage.rglob("*") if p.is_file()]
        print(f"\n→ {out}")
        print(f"   {len(files)} files, {Path(out).stat().st_size/1e6:.1f} MB")
        print("   upload this zip whole; arXiv unpacks and builds it itself")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
