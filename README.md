# How far affective audio models travel, and what adaptation recovers

Analysis code, run records, figure sources and manuscript for a study of how far
a mapping from acoustic descriptors to human affective response survives four
changes of setting — new material, edited audio, a sensor in place of a
self-report, and an individual listener — and of what each adaptation route we
tested recovers when it does not.

> **Manuscript source** · [`manuscript/`](manuscript/) — prepared for IEEE
> Transactions on Affective Computing
> **Preprint** · [arXiv:2608.27674](https://arxiv.org/abs/2608.27674) *(superseded — see the revision note below)*
> **Archive** · [10.5281/zenodo.23044439](https://doi.org/10.5281/zenodo.23044439) — concept DOI, always resolves to the latest version

---

## ⚠️ This supersedes an earlier preprint

An earlier version argued that generalisation failures divide into two kinds —
budget-limited and information-limited — with mutually exclusive remedies.
**That claim is contradicted by this project's own measurements and is
withdrawn.** The few-shot learning curve that priced the recoverable gap was
computed on the *cross-domain* corpus pairs, while the prose attributed it to
same-domain transfer and simultaneously described the cross-domain boundary as
unpurchasable. The figure legend had the condition right; the prose did not.

Two findings of independent value came out of the pre-submission audit, and
both concern corpora in wide use:

- **Emo-Soundscapes.** 613 of its 1,213 clips are mixtures of **30 source
  recordings**, occupying only 161 distinct constituent sets — a mean of 3.8
  near-duplicate clips per set. The distributed metadata gives every clip its
  own group identifier, so `GroupKFold` on the supplied groups is
  arithmetically identical to `KFold`. Evaluating on the 600 original
  recordings instead moves the within-corpus reference from 0.744 to 0.646 and
  median cross-domain transfer from 0.043 to 0.129.
- **PMEmo.** A sham using **presentation position alone** reproduces 114–249 %
  of the stimulus-level reliability of its song-level electrodermal measures,
  and song-to-position assignment is significantly non-random (ICC 0.041,
  p = 0.005). Those measures cannot serve as a stimulus-driven ceiling without
  a position control.

Everything withdrawn, corrected, recomputed and left standing is itemised in
[`reports/TAC_revision_audit.md`](reports/TAC_revision_audit.md). The superseded
preprint source is kept under `manuscript/archive/`.

---

## What the paper reports

For each of four boundaries: a within-corpus reference, the fraction that
survives an unadapted crossing, and what each adaptation route recovered —
with three evidence states kept apart throughout.

| state | meaning |
|---|---|
| **improvement observed** | a named route recovered a quantified fraction of the gap |
| **no improvement under the routes tested** | bounds those routes; *not* evidence that the information is absent |
| **insufficient evidence** | the design cannot separate the competing explanations, and we say which |

The practical consequence is about reporting convention. A zero-shot
cross-corpus number is an upper bound on the generalisation problem, not an
estimate of it: on the boundary we could price, an unadapted loss of 75 % falls
to roughly 33 % with a hundred target labels and to 25 % with none at all under
a second-moment alignment.

---

## What is in here

```
manuscript/      LaTeX source (IEEEtran), compiled PDFs
  archive/       the superseded preprint source
figures_r/       R sources for every figure; build_all.R builds them all
scripts/         analysis scripts, numbered in the order they were written
src/soundml/     shared library: features, modelling, provenance
reports/
  TAC_revision_audit.md   what was withdrawn, corrected and recomputed, and why
                          -- the current authority, start here
  PLAN.md                 superseded plan, kept as history; its two-kind
                          framework is withdrawn
  RESULTS_SUMMARY.md      findings, with the methodological ones kept separate
  FIGURE_LEGENDS.md       the single source of truth for figure descriptions
  source_data/            one machine-readable table per figure panel
  figures_r/              rendered figures — PDF / SVG / TIFF / PNG
runs/            one record per analysis run: script, seed, parameters, metrics
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

The analysis scripts expect the feature tables under `<repo>/data`. Those are
not redistributable and are not in this repository; if you have them elsewhere,
point `SOUNDML_DATA` at them:

```bash
SOUNDML_DATA=/path/to/data .venv/Scripts/python scripts/A9_tac_recheck.py
```

`A9_tac_recheck.py` and `A10_tac_recheck_embeddings.py` are the recomputations
behind the revision. Both reproduce every published value before varying
anything, so the two variants are directly comparable.

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

`A5` now also checks the **condition** each number belongs to, not only its
value. That is the check that would have caught the misattribution described
above: the number was numerically right and conditionally wrong, and a
value-only comparison passed it for months.

Each prints what it *cannot* check. None replaces looking at the figures — that
is how the last three real errors were found.

---

## What the run records do and do not establish

They fix the script, the seed, the parameters and the resulting metrics. They do
**not** pin the code: the records were written from working trees with
uncommitted changes, input hashes are present for a minority of runs, and output
hashes are absent throughout. The analyses reproduce from the scripts at the
recorded parameters; they do not yet reproduce bit-for-bit from a pinned commit.
The manuscript says the same. We would rather state the weaker guarantee than
let it be read as the stronger one.

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

Cite the manuscript, not the superseded preprint (arXiv:2608.27674), whose
central claim this work withdraws.

For the code and data, cite the Zenodo archive rather than the repository URL —
a URL says nothing about *which* state of the code you used.

| | DOI | when |
|---|---|---|
| **Concept** | [10.5281/zenodo.23044439](https://doi.org/10.5281/zenodo.23044439) | default — always resolves to the latest version |
| Version | [10.5281/zenodo.23044440](https://doi.org/10.5281/zenodo.23044440) | only when you depend on v1.0.0 specifically |

The badge Zenodo shows on the repository settings page is the *version* DOI, so
it is easy to pick up the wrong one.
