# Where affective audio models stop working, and what it costs to fix

Analysis code, run records, figure sources and manuscript for a study of how far
a mapping from acoustic descriptors to human affective response survives four
changes of setting — new material, edited audio, a sensor in place of a
self-report, and an individual listener — and what each crossing costs to buy
back.

> **Preprint** · arXiv:XXXX.XXXXX *(to be inserted at v1)*
> **Archive** · Zenodo DOI *(to be inserted at release)*
> **Manuscript source** · [`manuscript/`](manuscript/) · **Plan, including every withdrawn result** · [`reports/PLAN.md`](reports/PLAN.md)

---

## The finding in one table

"The model does not generalise" is not one diagnosis but two, and they call for
opposite remedies:

| | what fixes it | measured here |
|---|---|---|
| **budget-limited** | more target-side observations | swapping corpora inside a domain — 100 target labels return two-thirds of the gap |
| **information-limited** | a different information source; more data does not help | the response-channel crossing — no route exceeds 31% of the attainable ceiling |

Confusing the second for the first is the expensive mistake. The paper's purpose
is to price each boundary so the two can be told apart before the budget is
spent.

The dichotomy was declared falsifiable in advance — a gap closable *both* by more
observations *and* by a better representation would break it — and the paper
reports the case where it came closest to breaking.

---

## What is in here

```
manuscript/      LaTeX source, figure legends, compiled PDFs
figures_r/       R sources for all 16 figures (main + Extended Data)
scripts/         analysis scripts, numbered in the order they were written
src/soundml/     shared library: features, modelling, provenance
reports/
  PLAN.md              the operative plan, including every withdrawn result
  RESULTS_SUMMARY.md   findings, with the methodological ones kept separate
  FIGURE_LEGENDS.md    legends as they appear in the manuscript
  source_data/         one machine-readable table per figure panel (92 files)
  figures_r/           rendered figures — PDF / SVG / TIFF / PNG
runs/            one record per analysis run: script, seed, parameters, metrics
outreach/        drafts for data still being requested (IADS-E, Freesound API)
DATA_LICENSES.md      data licence ledger — read before redistributing anything
```

**Deliberately absent: the raw corpora.** They are ~38 GB across nine datasets,
several carry non-commercial terms, and one has no audio redistribution right at
all. See [`DATA_LICENSES.md`](DATA_LICENSES.md), which also records the four sets
of pretrained weights — one of those is non-commercial too. Every figure
regenerates from the source-data tables without any of it.

---

## Reproducing

```bash
uv sync
```

```bash
cd figures_r && Rscript build_all.R
```

`build_all.R` runs each figure script in a fresh environment and does not stop at
the first failure. Each script ends in assertions on the numbers the manuscript
quotes from it, so a figure that drifts from the prose breaks the build instead
of shipping a caption that disagrees with its own panel.

Re-running an analysis from the raw audio requires the corpora themselves;
acquisition routes and licence terms for all nine are in `DATA_LICENSES.md`.

### Consistency checks

```bash
.venv/Scripts/python scripts/A2_tex_structure_check.py
```

```bash
.venv/Scripts/python scripts/A3_artefact_currency.py
```

```bash
.venv/Scripts/python scripts/A5_figure_text_agreement.py
```

```bash
.venv/Scripts/python scripts/A4_withdrawn_number_sweep.py
```

In order: LaTeX structure and unreplaced placeholders; whether every artefact is
newer than the sources it was built from; whether figures and prose still agree,
recomputed from the CSVs the figures actually read; and whether any withdrawn
number survives anywhere in the text or the R sources.

Each prints what it *cannot* check. None replaces looking at the figures — that
is how the last three real errors were found.

---

## What the run records do and do not establish

They fix the script, the seed, the parameters and the resulting metrics. They do
**not** pin the code: the records were written from working trees with
uncommitted changes, input hashes are present for a minority of runs, and output
hashes are absent throughout. The analyses reproduce from the scripts at the
recorded parameters; they do not yet reproduce bit-for-bit from a pinned commit.
The Methods say the same. We would rather state the weaker guarantee than let it
be read as the stronger one.

---

## Licence

Code is MIT ([`LICENSE`](LICENSE)). Manuscript text and figures are CC BY 4.0.
The tables under `reports/source_data/` are published so the figures can be
verified without the corpora. Most are aggregate statistics; a few are complete
per-item feature tables, and for those the upstream corpus terms govern — the
per-corpus answer is the "derivatives redistributable" column of
[`DATA_LICENSES.md`](DATA_LICENSES.md).

**The data are not ours to license.** Nine corpora with nine different sets of
terms are listed with their conditions — including whether *derived* files may be
redistributed, which is often a different answer from the one for the audio — in
[`DATA_LICENSES.md`](DATA_LICENSES.md). Read it before redistributing anything built from
them, feature tables included.

## Citing

Cite the preprint. If you depend on a specific state of the code, cite the Zenodo
DOI for that release rather than the repository URL.
