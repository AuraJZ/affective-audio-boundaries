# Pre-submission revision audit — IEEE Transactions on Affective Computing

**Date:** 2026-09-29
**Manuscript:** `manuscript/main.tex` (+ `manuscript/supplementary.tex`)
**Preprint superseded:** `manuscript/archive/main_preprint_2026-08.tex`
**Constraint in force:** no new datasets, no new participants, no new listener
ratings. Everything below is re-analysis of data already in the project, or a
check against published sources.

---

## 0. The short version

The preprint's organising claim was that generalisation failures divide into
two kinds — *budget-limited*, which target-side observations close, and
*information-limited*, which they do not — with mutually exclusive remedies,
and that the domain boundary and the physiological boundary are of the second
kind.

**That claim is contradicted by the project's own measurements.** The few-shot
learning curve that priced the recoverable gap was computed on the
*cross-domain* corpus pairs; the prose attributed it to same-domain transfer
while simultaneously describing the cross-domain boundary as unpurchasable.
The figure legend stated the correct condition throughout. The prose did not.

The framework is withdrawn. What remains is a measurement paper: for each of
four boundaries, a within-corpus reference, what survives an unadapted
crossing, and what each adaptation route recovered — with an explicit ledger of
what the evidence cannot decide. The numbers are stronger for having lost the
framework, because two of the corrections move in the paper's favour
(§2.3, §2.4).

Two findings of independent value to the community came out of the audit, and
both concern corpora in wide use: a construction-leakage hazard in
Emo-Soundscapes (§2.1) and a presentation-order confound in PMEmo's
electrodermal reliability (§2.5).

---

## 1. How the audit was run

- **Recomputation.** Two new scripts, `scripts/A9_tac_recheck.py` and
  `scripts/A10_tac_recheck_embeddings.py`, recompute the corpus boundary from
  the feature tables using the project's own model registry
  (`src/soundml/regression.py`), so the recomputed values are directly
  comparable with the published ones. Both first **reproduce** every published
  number before varying anything; see §3.
- **Adversarial review.** A multi-agent audit covered eleven dimensions
  (learning-curve attribution, the dichotomy, the intervention experiment,
  individualisation, generalisation scope, leakage and protocol, physiology,
  references, TAC compliance, title/abstract/summary figure, supplementary
  consistency). 92 non-trivial findings were independently re-verified against
  the files and the data: **19 confirmed, 59 partially confirmed, 14 refuted.**
  Several of the refutations are of findings that had already been fixed while
  the verification was running, not of findings that were wrong.
- **Literature verification.** Every new citation was resolved against Crossref
  plus a second source before being added. Three proposed keys turned out to
  name works that do not exist as described and were replaced with the real
  ones (§6).
- **Journal requirements** were taken from the official TAC and IEEE Computer
  Society author pages, fetched during the audit, not from memory (§7).

---

## 2. What was withdrawn, corrected, or added

### 2.1 Emo-Soundscapes contains mixtures of thirty recordings (new finding)

**What.** 613 of the 1,213 Emo-Soundscapes clips are mixtures built from five
selected clips in each of six Schafer categories — **30 source recordings** —
combined two or three at a time at two relative levels. Parsing the distributed
filenames, which encode the constituents, shows the 613 mixtures occupy only
**161 distinct constituent sets**, a mean of 3.81 clips per set and up to 8.
The label table gives every clip its own `group`, so `GroupKFold` on this corpus
is arithmetically identical to `KFold`.

**Why it matters.** The Methods of the preprint — and of other work using this
corpus — describe grouped cross-validation as preventing excerpts of one
recording from spanning folds. On this corpus it does not.

**Effect, recomputed on the 600 originals alone:**

| quantity | as distributed | 600 originals |
|---|---|---|
| Emo-Soundscapes within-corpus ρ (median of 6 algorithms) | 0.744 | **0.646** |
| cross-domain transfer ρ (median over 6 pairs × 6 algorithms) | 0.043 | **0.129** |
| cross-domain transfer ρ (mean, s.d.) | 0.058 (0.155) | 0.115 (0.153) |
| cross-domain loss vs. target's own reference | 93.3 % | **74.9 %** |
| same-domain loss | 15.0 % | 15.0 % |

Both changes shrink the apparent domain gap: the leakage inflated the corpus's
own ceiling *and* depressed transfer into it, because a test set half composed
of near-duplicates distorts the ranking a rank correlation is computed on.

**In the manuscript:** §III-C1 and Table I (`tab:hazard`); Table III
(`tab:families`) reports every representation on both variants; the Discussion
carries it as a reuse hazard.

### 2.2 The learning curve was attributed to the wrong condition (blocking)

**What.** `figures_r/fig_price.R` filtered `cross_domain`, and
`scripts/A5_figure_text_agreement.py` did the same. The published curve —
20/34/50/64/69 % at k = 10/25/50/100/200 — is therefore the **music ↔
environmental** curve. The preprint's abstract (l. 24–25) and Discussion
(l. 791–793, 884–885) attributed it to transfer "to a new corpus of the same
kind" / "within a domain", while the same Discussion classified the
cross-domain boundary as one that is "not for sale".

**Consequence.** The boundary the preprint called unpurchasable is precisely the
one whose price it quoted. The dichotomy fails at boundary 1 by the preprint's
own stated falsification condition.

**Corrected, on the leak-free corpus (median over six ordered pairs of each
kind, six-algorithm denominator):**

| k | same domain | across domains |
|---|---|---|
| 10 | 8.6 % | 18.8 % |
| 25 | 16.0 % | 30.9 % |
| 50 | 33.9 % | 43.5 % |
| 100 | **49.9 %** | **56.9 %** |
| 200 | 76.5 % | 71.0 % |

**In the manuscript:** §IV-B2 states the correction explicitly rather than
silently substituting numbers; Table II (`tab:adaptation`) gives both
conditions and both denominators; Fig. 2a plots both curves.

### 2.3 The CORAL sham control was invalid on half the pairs, in the direction that understated the result

**What.** `scripts/80_domain_adaptation.py` says so in its own comments: with
one environmental corpus, when the source is environmental the "unrelated third
corpus" is necessarily music and therefore in the **target's** domain. The
preprint pooled the valid and invalid halves and reported a 14 % sham,
discounting CORAL's 22 % gain to about a third of face value.

**Split by validity (clean corpus, cross-domain):** valid sham **−11.1 %**,
invalid sham **+25.1 %**. The entire pooled sham effect is the invalid half.
CORAL's gain (25.0 %) is therefore *more* target-specific than the preprint
claimed, not less. The decomposition the script recommends instead confirms it:
per-dimension variance alone recovers 4.9 %, and the further 20 points — 80 %
of the total — come from matching correlation structure.

**In the manuscript:** §IV-B2, Table II, Fig. 2b and 2c.

### 2.4 "The return per doubling collapses after 100 labels" does not survive

On the leak-free corpus the increments are 10, 14, 10 and 14 points — no
collapse, and the largest increment is the last. The reported collapse was
produced jointly by the mixture leakage and by the loss of the Soundtracks
pairs at k = 200 (the protocol excludes k ≥ n/2). Withdrawn, and with it any
claim about where the curve saturates.

### 2.5 PMEmo electrodermal reliability is reproduced by presentation position (new finding)

**What.** `scripts/94_order_confound.py` ran this check and its output sat in
`reports/source_data/order_confound.csv` unreported.

| measure | true labels | position-only sham | sham as % of true | position partialled out |
|---|---|---|---|---|
| skin-conductance response rate | 0.116 | 0.290 | **249 %** | 0.038 |
| phasic mean | 0.237 | 0.283 | 120 % | 0.081 |
| skin-conductance level slope | 0.234 | 0.267 | 114 % | 0.206 |

Song-to-position assignment is itself significantly non-random (ICC 0.041,
p = 0.005).

**Consequence.** The preprint expressed the cross-response-channel result as a
percentage of an attainable ceiling — 31 %, 24.5 %, 21.3 % — where the ceiling
was √r of one of these reliabilities. **That framing is withdrawn in full.** A
denominator a position sham reproduces cannot bound a stimulus-driven
predictor. Two further defects in the same passage: the percentages selected
across denominators differing by 40 % between measures, in a direction that
made the *lowest*-reliability measure yield the *highest* percentage; and the
assertion that "their confidence intervals overlap" refers to intervals that
were never computed for the acoustic routes.

**What survives:** the raw comparison. No acoustic route to these electrodermal
measures exceeds a random projection of the same descriptor space, and the
listeners' own ratings do, weakly (ρ = 0.117, p = 0.004).

**In the manuscript:** §III-C2, §IV-E3, and Supplementary §S3.

### 2.6 "The barrier is calibration, not sign" is contradicted by its own data

**What.** The reported sign agreement pools, for each model, its 40
training-domain excerpts with 40 from the other domain. Recomputed from
`runs/20260727T100401Z_counterfactual_intervention/counterfactual_raw.parquet`:

| model | material | n | median slope | Wilcoxon p |
|---|---|---|---|---|
| music | pooled (as reported) | 80 | +0.015 | 3×10⁻⁴ |
| music | music excerpts | 40 | +0.099 | 2×10⁻⁴ |
| music | environmental | 40 | +0.012 | 7×10⁻⁴ |
| environmental | pooled (as reported) | 80 | +0.074 | <10⁻⁴ |
| environmental | music excerpts | 40 | +0.080 | <10⁻⁴ |
| environmental | **environmental** | 40 | **−0.019** | **0.073** |

The environmental model shows the effect on *music* excerpts and not on
environmental ones — the material it was fitted on, and the material the claim
is about.

**Independently, the claim was incompatible with the metric.** Every transfer
number in the paper is a Spearman rank correlation, which is exactly invariant
under any strictly monotone recalibration. A purely calibrational deficit
cannot move ρ at all.

**Also corrected:** sign consistency is computed against the sign of the
observed median, so 50 % is its arithmetic floor and not a chance level (null
median 0.54). Values of 57–60 % are at the floor. And the perturbed models are
binary random-forest classifiers, not the regression models of Fig. 1 — a
distinction the preprint did not make.

**In the manuscript:** §IV-D1 and Table IV (`tab:intervention`).

### 2.7 The out-of-sample personalisation test existed and was not reported (blocking)

`reports/source_data/pmemo_personalisation_gain.csv` and
`subjective_individual_dual.csv` contain a direct test of the question the
break-even framing asks. Neither is cited anywhere in the preprint.

- **PMEmo electrodermal:** per-person slopes lose to the population slope in
  **15 of 15 cells** (−0.109 to −0.072 at 18 observations per person, q = 1.0).
- **DEAM ratings:** negative in **all 10 cells** at n = 18, 30 and 50
  (−0.125 to −0.052, −0.054 to −0.020, −0.028 to −0.002); at n = 100, four of
  ten turn positive with the best at p = 0.054.

The trajectory is consistent with the variance model's n\* = 166 for that
setting and to that extent validates the extrapolation's direction. It also
settles the applied question the other way from the preprint: at every
observation count these corpora supply, per-person calibration measurably
loses.

**No such test exists for ARAUS**, the setting with the lowest n\* and the only
one where the model implies break-even has been reached.

**In the manuscript:** §IV-F3 and Table VI (`tab:individual`), which carries the
out-of-sample gain as a column beside n\*.

### 2.8 n\* = 318 is a median over an undisclosed subset

Ten of the fifteen electrodermal cells have a corrected between-person spread of
exactly zero and therefore an infinite n\*. The reported 318 is the median over
the five cells where n\* is finite — one of which does not exceed its own null.
The median over all fifteen is infinite; over the four that do exceed their null
it is 264. All three are now given and none is presented as a price.

### 2.9 The two ARAUS cells that cross break-even sit on an axis with no passing control

`reports/source_data/araus_individual.csv` records that the corpus's own
sound-to-masker-ratio manipulation check passes for pleasantness
(p = 4×10⁻⁶) and **fails for eventfulness** (p = 0.38). Both crossing cells are
on the eventfulness axis. Stated in §IV-F3 and in the Fig. 6 description.

### 2.10 The three per-person settings are not comparable

They differ in corpus, descriptor set *and* outcome axis simultaneously: ARAUS
uses ISO-532 psychoacoustic descriptors against ISO soundscape axes; DEAM uses
spectro-temporal and tonal descriptors against valence and arousal; PMEmo uses
those same descriptors against electrodermal measures. The preprint's conclusion
that "choosing the wrong corpus is as expensive as choosing the wrong channel"
is withdrawn.

### 2.11 Units of replication were overstated sixfold

"36 ordered music-to-music pairs" is 6 ordered corpus pairs × 6 algorithms.
There are six pairs of each kind. Corrected in §IV-B1 and in the threats section.

### 2.12 All claims of prior specification removed

No traceable artifact predates the corresponding analyses for any criterion
described as "fixed in advance", "pre-declared" or "falsifiable in advance":
the manuscript directory was untracked in version control, the repository head
predates most run records, and only 2 of 199 run records carry a declared
hypothesis. Every such phrase is removed from the manuscript, the supplement
and the figure descriptions, and nothing is presented as preregistered.

### 2.13 Smaller corrections

- "The same 1,240 trials" for both the EEG and self-report reliability sets is
  false: EEG uses 1,240 trials over 307 stimuli, self-report 1,080–1,121 over
  285–298.
- The EEG intraclass correlation is the maximum of 156 candidates and the
  self-report value the maximum of 8, an asymmetry in the direction favouring
  the EEG side. Now stated.
- CLAP pretraining overlap is described as possible at the corpus level, not
  verified: no sample-level overlap check exists in the repository.
- The Wav2Vec2 checkpoint is described by its pretraining *domain* (speech), not
  claimed to be purely self-supervised.
- Emo-Soundscapes was annotated by pairwise comparison, not direct rating; this
  is now offered as a competing explanation for the domain gap alongside sound
  domain itself.
- ds002721's stimuli come from the same published stimulus set as Soundtracks;
  the two corpora are not independent material.
- Table 1's "Hand-specified (161)" contradicted the text's 122. The descriptor
  set used for the transfer analysis is 122; corrected.
- The four-boundary summary figure is withdrawn rather than corrected, for three
  independent reasons recorded in the legend source.

---

## 3. What was recomputed, and what was not

### Recomputed (and reproducing the published values first)

`scripts/A9_tac_recheck.py` reproduces, on the corpus as distributed:

| quantity | published | recomputed |
|---|---|---|
| within-corpus medians (DEAM / PMEmo / Soundtracks / Emo) | 0.725 / 0.801 / 0.810 / 0.744 | 0.724 / 0.801 / 0.810 / 0.744 |
| same-domain transfer, mean (s.d.) | 0.592 (0.130) | 0.591 (0.131) |
| cross-domain transfer, mean (s.d.) | 0.054 (0.161) | 0.058 (0.155) |
| few-shot recovery at k = 100, cross-domain | 64 % | 64.5 % |

`scripts/A10_tac_recheck_embeddings.py` reproduces Table 1's five representation
rows exactly on the published corpus, then recomputes them on the leak-free one.
Outputs: `reports/source_data/tac_recheck_{emo_structure, within_corpus,
transfer, fewshot, alignment, table1}.csv`.

Also recomputed: the intervention disaggregation (from the run's raw parquet);
the n\* cell tables (from `canonical_tau_table.csv`); the order-confound sham
(from `order_confound.csv`); the out-of-sample personalisation gains.

### Not recomputed

- **The 84.4 % within-corpus figure.** Unchanged. PMEmo has one clip per song,
  so no construction leakage is possible there, and the reliability estimate
  and the model performance are on the same excerpts. Its three stated
  optimism sources (winner's bias over six algorithms, a bootstrap that does
  not refit, a reliability carried as fixed) are unchanged and restated.
- **The EEG analyses of ds002721.** The controls, the 156-measure reliability
  comparison and the individual-level analysis were re-read and re-scoped but
  not re-run.
- **The ds003690 design requirement.** Not re-run; it is the strongest
  uncontaminated result in the paper and nothing in the audit bears on it.
- **The tonality result.** Not re-run; only its framing changed.
- **The ARAUS τ estimates.** Not re-run. The corpus's own corrigendum could not
  be read (§6), so we did not attempt to re-derive anything from it.

---

## 4. Evidence for each core claim as it now stands

| claim in the manuscript | evidence | state |
|---|---|---|
| Within-corpus prediction reaches 84.4 % of the annotation-reliability ceiling | PMEmo, ρ = 0.817 best of six, split-half r = 0.935, √r correction, bootstrap CI | measured; upper bound, three stated optimism sources |
| Unadapted same-domain corpus swap costs ~15 % | 6 ordered pairs × 6 algorithms, per-pair against the target's own reference | measured |
| Unadapted cross-domain swap costs ~75 % | same, leak-free environmental corpus | measured |
| 100 target labels recover ~50 % (same domain) / ~57 % (across) | ridge, 20 resamples per cell, both denominators reported | measured |
| Label-free CORAL recovers 25 % of the cross-domain gap, target-specifically | valid sham −11 %, variance/correlation decomposition | measured |
| Four pretrained representations recover little of the cross-domain gap | Table III, leak-free; best improves on hand-specified by 0.034 | measured; bounds the four tested, not all representations |
| A targeted representation change recovers perceived happiness | +0.132 vs +0.045; tonal alone 0.580 > spectro-temporal 0.474; mechanism check dissociates | measured |
| Editing audio moves model output, inconsistently across model and material | 2×2 disaggregation, Table IV | measured; model output only, no listener |
| No acoustic route to song-level EDA exceeds a random-projection control | ρ = 0.014–0.070, p = 0.03–0.34 | measured; **no ceiling quotable** |
| EEG carries less than half the stimulus-specific structure the ratings do | max ICC 0.092 [0.037, 0.142] vs 0.221 [0.155, 0.281] | bound, not absence; unequal selection bias stated |
| Per-person detection needs 57–233 observations depending on channel | ds003690, 75 listeners, out-of-sample validated extrapolation | measured |
| Per-person calibration is not shown to pay | n\* = 67–∞; out-of-sample gain negative at every available budget | model extrapolation + direct negative test |

---

## 5. Limitations that remain, and where they appear

| limitation | where stated |
|---|---|
| One environmental corpus: domain and dataset are not separable, and the corpus also differs in annotation method | §III-A, §V-C, Fig. 1 description |
| 36 transfer values are 6 corpus pairs × 6 algorithms | §IV-B1, §V-C |
| ds002721 and Soundtracks share a stimulus pool | §III-F, §V-C |
| Selection: 84.4 % is best-of-six; EEG ICC is max-of-156 against max-of-8 | §IV-A, §IV-E2, §V-C |
| No multiplicity control across 33 per-person cells; 60 bootstrap replicates per null; two-stage slope estimation | §V-C |
| No out-of-sample personalisation test for ARAUS; its crossing cells are on an axis whose control fails | §IV-F3 |
| No target-side learning curve on a physiological outcome | §IV-E3, §V-D |
| The PMEmo electrodermal pipeline has no passing positive control | §III-F |
| Nothing is preregistered | §III-H, §V-C |
| Reproducibility: uncommitted trees, input hashes for 38 of 199 records, no output hashes, no test suite | §VI Declarations |
| The ARAUS corrigendum's scope could not be established | §III-G |
| Nothing here bears on sleep or clinical efficacy | §V-C, Declarations |

---

## 6. References

All 32 pre-existing entries were previously verified and are unchanged.
Twenty-three entries were added, each resolved against Crossref plus a second
source. Three requested keys named works that do not exist as described:

- a 2018 Zhang cross-corpus paper → the real one is **Zhang, Provost & Essl,
  IEEE TAFFC 10(1), 2019** (the 2017 DOI suffix is the online-first year);
- a 2012 Yang cross-dataset MER paper → the canonical work is **Hu & Yang,
  IEEE TAFFC 8(2), 2017**;
- "Yang 2007, Personalized music emotion recognition" → the 2007 paper is
  titled *the role of individuality*; the paper actually called *Personalized
  music emotion recognition* is a 2009 SIGIR poster. Both are now cited.

Gaps closed: AST, MERT, Wav2Vec2 and AudioSet had **no bibliography entry at
all** although they carry Table III; CORAL and subspace alignment were
unattributed; Spearman–Brown and the attenuation correction were uncited.

**Daly et al. 2015** (*Brain and Cognition* 101:1–11) analyses the same
recordings as ds002721 and reports that music-induced emotion can be predicted
from brain activity *combined with* acoustic features at r = 0.234 (p < 0.001),
with either alone significantly worse. It is now cited and positioned: it is a
group-level, within-dataset, two-modality prediction result, whereas ours is a
stimulus-level reliability estimate on EEG measures alone. The two are
compatible. (Note: that paper's abstract also claims "over 20 % of the
variance", which is not consistent with r = 0.234; we do not quote it.)

**The ARAUS corrigendum** (*IEEE TAFFC* 16(2):1260, doi
10.1109/TAFFC.2024.3435997) is cited alongside the dataset paper. **Obtained
and read 2026-09-29; the question is closed.** It corrects the author list of
citation [90] in the ARAUS paper — attributed there to Gamst, Meyers, Burke and
Guarino, actually Watson, Clark and Tellegen (the PANAS scales paper). The
corrigendum notes that the DOI and full-text links in the published version
already resolved to the correct work, and states that "the error appears to be
purely typographical in nature and does not affect the study's findings and
conclusions".

It touches nothing this paper uses: not the psychoacoustic columns
(`Navg_r`, `Savg_r`, `Ravg_r`, `Favg_r`), not the participant ratings, and not
the per-person calibration estimates derived from them. The caveat paragraph
the manuscript carried while this was unverified has been removed rather than
left as standing doubt.

Two indirect signals had pointed the same way beforehand and are recorded here
because they are cheap and were right: the ARAUS data repository shows no
erratum or dataset re-release, and the arXiv version history of the ARAUS
preprint (2207.01078) gives v4, dated 2 July 2024, the comment "Fixed
inaccurate author list in citation #90" at an identical file size to v3.

---

## 7. TAC compliance

Requirements taken from the official pages during the audit:

- **Page limits.** TAC's author page states: *"Regular paper — 12 double column
  pages (Submissions may be up to 18 pages in length, subject to MOPC.)"*
  Twelve is the **Mandatory Overlength Page Charge threshold** ($220 per page
  or fraction beyond it); **eighteen is the hard cap on what may be submitted.**
  The current manuscript is **18 pages**, which is inside the cap and implies
  six overlength pages at $220 = **$1,320**. That is a cost decision for the
  authors, not a compliance failure. Further compression is possible — the
  Methods already delegate to the uncapped supplement and more could follow.
- **Template.** IEEEtran journal class, `IEEEtran.bst`. Done; the bioRxiv
  preamble is archived.
- **Abstract.** 100–200 words, no math, no references. Rewritten from 439 words
  to **200**, with no math and no citations.
- **Index Terms.** Added.
- **Figures.** Now IEEE floats placed at their first callout, not after the
  bibliography. Eight figures reduced to six; the long Nature-style legends
  moved to Supplementary §S9, which is uncapped.
- **Supplementary numbering.** "Extended Data Figure N" → "Fig. SN" throughout,
  with visible caption labels.
- **Review model.** TAC is **single-anonymous** by default, so no masking of
  authors, affiliation, repository URL or competing interests is required.
- **Ethics.** Blanket statement replaced by a per-corpus table
  (Supplementary Table S1) that states, for each of nine corpora, what its own
  source reports — and says "not stated in source" where nothing is reported,
  rather than implying approval exists.

### Outstanding, and requiring an author decision

1. **Overlength charge.** 18 pages implies six pages beyond the MOPC threshold
   at $220 each. Either accept the $1,320 or compress further; the Methods
   already delegate to the uncapped supplement and more could follow.

### Settled by the authors

**ORCID.** J. Zhang's iD (0009-0008-8738-3783) is now set after the author name
as the linked ORCID icon (`orcidlink`). X. Yao has no ORCID, so that name
carries no icon — nothing is left blank and none was invented. ScholarOne will
still prompt for one during submission; registering an iD is free and takes a
few minutes, and doing it before submitting avoids a mid-submission stop.

**Generative-AI disclosure.** The authors supplied the final wording, which now
stands in the Declarations. It identifies the system, names the affected
sections, states what was and was not AI-generated, records the review the
authors performed, and assigns responsibility — which is what IEEE's policy
asks for. The section numbers in it are `\ref`s rather than literals so they
cannot drift if the document is reordered; they resolve to III-D, IV-B and
IV-D as written.

---

## 8. Verification performed on the revised manuscript

- `main.pdf` compiles under IEEEtran: **18 pages**, no LaTeX errors, no
  undefined references or citations, no overfull boxes above 15 pt.
- `supplementary.pdf` compiles: **46 pages**, no errors, no undefined citations.
- All 50 rendered bibliography entries were inspected, not just counted.
  One real defect was found this way, and one false alarm is worth recording
  alongside it because neither produced a LaTeX error and the document
  compiled with zero undefined citations throughout:
  1. **False alarm.** The ARAUS corrigendum [48] ends on two closing double
     quotes, because the published title contains the corrected article's
     title in quotes and `IEEEtran.bst` adds its own pair around the whole
     thing. This looks like a defect and is not: **IEEE's own citation export
     renders it identically** (`...Urban Soundscapes"" in IEEE Transactions on
     Affective Computing, vol. 16, no. 02, pp. 1260-1260`). It was briefly
     changed to single quotes on the assumption that the doubling was ours.
     Reverted — reproducing what the publisher prints is what a reviewer
     checking against Xplore expects. The bib entry now carries a note saying
     not to "fix" it again.
  2. **Real defect, introduced by that attempted fix.** Adding an explanatory
     `%` comment *inside* the entry silently broke it: BibTeX 0.99 rejects `%`
     within an entry and skips the remainder, degrading the reference to a bare
     author dash with no title, journal, volume or year — still with zero
     undefined citations. The `.blg` said so (`I'm skipping whatever remains of
     this entry`, plus three `empty field` warnings); the `.log` did not.
     **Read the `.blg`, not only the `.log`.** The comment now sits outside the
     entry, and both `.blg` files report zero warnings.
- All nine figure scripts rebuild and **every internal `stopifnot` assertion
  passes** (`figures_r/build_all.R`).
- `scripts/A5_figure_text_agreement.py` rewritten against the corrected numbers
  and extended with a **condition check** — the registry now records which
  experimental condition each number belongs to, and the check fails if the
  condition word does not appear near the number in the text. This is the check
  that would have caught the 2026-08 misattribution: the old registry compared
  values only, so a number that was numerically right and conditionally wrong
  passed. **19/19 registered checks pass, including all condition checks.**
- `scripts/63_legends_to_tex.py` updated for S-numbering, with an assertion
  against duplicate macro names — the previous regex silently skipped any block
  it could not parse, which would have dropped a figure description without
  warning.

## 9. What this audit did not do

- No new data of any kind.
- The EEG, ds003690 and tonality analyses were not re-run, only re-scoped.
- The ARAUS τ estimates were not re-derived.
- No attempt was made to obtain a second environmental corpus, which is the
  single change that would resolve the largest open question.
- Reproducibility was **not** upgraded: the run records still come from
  uncommitted working trees, input hashes cover 38 of 199 records, output
  hashes are absent and there is no test suite. The manuscript says so rather
  than claiming the stronger guarantee.
