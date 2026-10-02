# Motor Imagery Decoding with EEGNet — BCI Competition IV-2a

Undergraduate thesis project (TCC) on decoding motor imagery from EEG with a compact
convolutional network, aimed at driving a game in real time over TCP.

Implementation of [EEGNet (Lawhern et al., 2018)](https://doi.org/10.1088/1741-2552/aace8c)
in TensorFlow/Keras, evaluated on [BCI Competition IV, dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) (Graz).

> Notebooks are written in Portuguese — they were built as a guided course, with
> "think first" prompts, exercises and reflections. The code and this README are in English.

---

## Evaluation protocol

This matters more than the accuracy number, so it goes first.

**Within-subject, cross-session, subject A01 only.** The dataset ships each subject as
two files recorded on **different days**: `A01T.gdf` (training session) and `A01E.gdf`
(evaluation session, labels in a separate `.mat`). Training uses `A01T`, testing uses
`A01E`. This is the official BCI Competition protocol — there is no random train/test
split, and data augmentation is therefore structurally incapable of leaking across it.

These are **not cross-subject** results, and **not** an average over the nine subjects.
This section originally stated that A01 is one of the stronger performers in the dataset;
having now run the other eight, that is not supported -- A01 is a median subject. See the
nine-subject replication below.

| | Trials |
|---|---|
| Train (`A01T`), 2-class | 144 → 116 train / 28 validation → 696 after augmentation |
| Train (`A01T`), 4-class | 288 → 232 train / 56 validation → 1392 after augmentation |
| Test (`A01E`), 2-class | 144, untouched |
| Test (`A01E`), 4-class | 288, untouched |

Preprocessing: 22 EEG channels, IIR bandpass 4–40 Hz, epochs 0–4 s after cue, z-scored
per epoch (per-epoch statistics only, so no dataset-level statistic crosses the split).
Trials the recording marks as artifact-contaminated (event `1023`: 15 in `A01T`, 7 in
`A01E`) are **kept**, in both training and test — the usual convention on this dataset,
and dropping them from the test set would inflate the reported accuracy.

### Subject split for the nine-subject work (declared 2026-10-01)

Everything above this line is A01 only. The nine subjects are now on disk, so the work
that follows needs a development/test split over *subjects*, and it is declared here
before it is used.

**Development: A01. Test: A02–A09, locked.**

A01 is the development subject because it is already spent. As the Limitations section
documents, `A01E` was consulted at every design decision in the single-subject phase, so
nothing can make it clean again; naming it the development subject costs nothing that was
not already paid, and it leaves eight never-inspected subjects for the paired tests of the
transfer and mSEM phases. Every design decision from here — freezing policy, calibration
budget, domain alignment, architecture — is taken on A01 and on validation splits of each
`A0xT`, never on the A02–A09 evaluation sessions.

The zero-shot transfer table is the one exception, and it is not a real one: it is purely
descriptive and selects nothing. No variant was compared, kept or discarded on the basis
of it. The split above was fixed before those runs were executed.

This is a direct response to the methodological failure recorded in Limitations. It does
not retroactively fix the A01 numbers, and it is not claimed to.

## Results

Test set is the held-out `A01E` session. Every configuration was run with multiple random
seeds, because a single run is not a result — see Limitations.

| | seeds | accuracy (mean ± sd) | median | range | Cohen's κ (mean ± sd) |
|---|---|---|---|---|---|
| 2-class (left vs. right hand) | 5 | 82.6% ± 14.1% | **88.2%** | 57.6 – 92.4% | 0.653 ± 0.283 |
| 4-class (left, right, feet, tongue) | 5 | **73.9% ± 3.2%** | 72.9% | 70.8 – 78.5% | 0.652 ± 0.042 |

Chance is 50% and 25% respectively. Per-seed numbers, confusion matrices and full
classification reports are in `results_final.json`.

The **4-class model is far more stable than the 2-class one** (± 3.2% vs ± 14.1%), which
is the opposite of what task difficulty alone would predict. It has twice the training
data — 232 trials against 116 — and, just as importantly, a 56-trial validation set
instead of 28, so early stopping has a usable signal. Data quantity, not task difficulty,
is what governs reliability at this scale.

### Seed ensemble

Because run-to-run variance is the dominant error source, the primary result is an
ensemble: every seed's model is kept, and their softmax distributions are averaged
(soft voting). No member is ever selected or discarded using the test set.

| | members | median member | best single member* | soft vote | hard vote | ensemble κ | McNemar vs median member |
|---|---|---|---|---|---|---|---|
| 2-class | 5 | 88.2% | 92.4% | **91.0%** | 91.7% | 0.819 | p = 0.29 |
| 4-class | 5 | 72.9% | 78.5% | **79.2%** | 77.4% | 0.722 | **p = 0.013** |

\* not selectable in advance — you only know which member was best after seeing the test set.

Soft voting is reported as the primary result because it is the a-priori-defensible default
(it preserves each member's uncertainty instead of collapsing it to a label), not because it
scored higher — hard voting is in fact slightly better in 2-class and slightly worse in
4-class. Both are plotted in `fig4_ensemble_vs_members.png`.

For 4-class the ensemble is a clear win over the *typical* draw: +6.3 points over the median
member, and the paired McNemar test is significant (33 trials the ensemble alone gets right
against 15 for the median member, p = 0.013). It also edges past the best individual member,
but only by 79.2% against 78.5% — two trials out of 288, which is noise. The honest claim is
that the ensemble reliably lands near the top of the seed distribution without having to know
in advance which seed is good, **not** that it exceeds the best seed.

For 2-class the ensemble lands between the median member (88.2%) and the best (92.4%). The
improvement is in the expected direction but 144 test trials are not enough to establish it
(p = 0.29).

Mean pairwise disagreement between members is 0.231 (2-class) and 0.242 (4-class) — the
members make substantially different errors, which is the precondition for averaging to
help at all. The SE ablation below is a direct test of that precondition: it makes the
members more alike, and in 4-class the ensemble gets worse as a result.

A secondary variant discards members more than one standard deviation below the mean
*validation* accuracy — a rule computable without the test set, using the accuracy of the
model `EarlyStopping` actually restored, on that seed's own clean validation split. It drops
the collapsed seed in 2-class (→ 92.4%) and seed 2 in 4-class (→ 79.5%), improving both. It
is still reported as secondary: each seed validates on a different 28 (or 56) trials, so
those accuracies are not strictly comparable across seeds, and the all-members ensemble
needs no member-selection rule at all.

Full ensemble metrics are in `results_ensemble.json`, produced by `ensemble.py`.

Softmax confidence separates by correctness in both, which is what makes a rejection
threshold viable for online use (seed 0 models):

| | mean confidence, correct | mean confidence, incorrect |
|---|---|---|
| 2-class | 87.8% | 71.0% |
| 4-class | 75.8% | 63.8% |

### Relation to earlier numbers in this repo

The exploratory notebook reported 88.9% (2-class) and 80.2% (4-class) from single runs
with a contaminated validation split. Under the corrected protocol the 2-class figure
holds up — 88.2% median, and the best-performing seed reaches 92.4% — but the 4-class
figure does not: it settles around 74%. The earlier 80.2% benefited from broken early
stopping, which let that model train far longer than a clean validation signal would have
allowed. The notebook numbers should be read as superseded by the table above.

The same applies to the notebook's other headline claim, the +8.6 points from its
Squeeze-and-Excitation variant. Re-measured as a paired ablation it is worth no accuracy at
all, though it turned out to be doing something else that does hold up. See
[The Squeeze-and-Excitation block](#the-squeeze-and-excitation-block).

## Data augmentation

Additive Gaussian noise, σ = 0.1 scaled to each epoch's own standard deviation, five
copies per trial. It simulates the trial-to-trial variability of background EEG while
leaving the ERD/ERS modulation — a slow change in mu/beta power — intact.

```python
noise = rng.standard_normal(X.shape) * (0.1 * X.std(axis=(1, 2), keepdims=True))
```

### Does it actually help?

Ablation over the same five seeds per task: identical architecture, identical split,
identical batch size and patience — only the augmentation is switched off. The epoch cap is
raised in the no-augmentation arm (1200 / 600 instead of 300 / 150) because without the five
copies each epoch has six times fewer gradient steps; `EarlyStopping`, not the cap, ended
every single run (the longest was 263 epochs), so the comparison is not confounded by the cap.

| | with augmentation | without | Δ | seeds improved |
|---|---|---|---|---|
| 2-class | 82.6% ± 14.1% | 65.3% ± 17.2% | **+17.4 pts** | 5 / 5 |
| 4-class | 73.9% ± 3.2% | 61.2% ± 19.8% | **+12.7 pts** | 5 / 5 |

Augmentation helps **every seed in both tasks** — there is no seed where it hurts. The
effect is not a uniform shift: it mostly consists of preventing collapses. Without it three
of five 2-class seeds land at or near chance (50.7%, 52.8%, 54.9%) and one 4-class seed
collapses to 26.0%, while the seeds that already worked gain only 1–8 points. Read that way,
the augmentation is buying *reliability*, which is the same axis the seed ensemble attacks.

Per-seed numbers are in `results_ablation.json`; the runs themselves in `runs/ablation/`.

**The validation split happens before augmentation.** This is the one thing worth copying
from this repo. Keras' `validation_split=0.2` reserves the *last 20% of the array without
shuffling*; on an array built as `[originals, copy1, ..., copy5]` those last rows are noisy
duplicates of trials already in training. The symptom is unmistakable — validation accuracy
pinned at 1.000 while test accuracy sits far below — and it makes `EarlyStopping` select
epochs on a meaningless signal. It never contaminated the test set here (that is a separate
session file), but it did have to be fixed. In `train_final.py` the order is explicit:
`stratified_split()` picks the validation indices from the *original* trials, and
`augment_gaussian_noise()` is then applied to the training part only.

## The Squeeze-and-Excitation block

The course notebook's Exercise 8 asked for an architecture variant of my own. I added a
Squeeze-and-Excitation gate over the spatial filters: global-average-pool each of the 16
spatial feature maps to one number, pass those through a 16 → 4 → 16 bottleneck, and use
the resulting sigmoid as a per-filter multiplier. The network gets to say which spatial
filters matter for a given trial instead of weighting all 16 equally. It sits immediately
after the ELU of `spatial_conv` and before the first `AveragePooling`, and it costs **148
parameters** (3018 → 3166 in 2-class, 4012 → 4160 in 4-class, so +4.9% and +3.7%).

In the notebook it took 4-class from 70.7% to 79.3%, and I recorded that as +8.6 points.
**That number does not survive this repo's own standards**, for three independent reasons:
it was `max(val_accuracy)` rather than the restored model, which is optimistic by up to 10
points here; it was a random 80/20 split *within* `A01T`, not the held-out `A01E` session;
and it was one run against one run, on tasks where the seed spread reaches 34.8 points.

So it was rebuilt as a paired ablation in `se_ablation.py`: identical architecture,
identical split, identical augmentation, identical batch size and patience, identical test
session, five seeds per task, pairing seed-for-seed against the existing `runs/run_*.json`.
Only the SE block is switched on.

| | base | + SE | Δ mean | Δ median | seeds improved |
|---|---|---|---|---|---|
| 2-class | 82.6% ± 14.1% | **87.1% ± 2.4%** | +4.4 pts | −0.7 pts | 2 / 5 |
| 4-class | 73.9% ± 3.2% | **74.4% ± 0.9%** | +0.6 pts | +1.7 pts | 3 / 5 |
| κ, 2-class | 0.653 ± 0.283 | 0.742 ± 0.049 | | | |
| κ, 4-class | 0.652 ± 0.042 | 0.659 ± 0.013 | | | |

**There is no accuracy gain.** Paired Wilcoxon gives p = 1.00 (2-class) and p = 0.81
(4-class); the sign test gives p = 1.00 for both. The +8.6 points of the notebook do not
reproduce as an effect of the architecture, and the 2-class *median* actually drops.

**There is a large variance reduction, and it is the only effect here that reaches
significance.** The across-seed standard deviation falls by a factor of 33.7 in 2-class and
11.4 in 4-class. Under the Pitman–Morgan test, which is the appropriate test for variances
of *paired* samples, p = 0.016 and p = 0.030.

The block compresses both tails, in both tasks:

| | worst seed | best seed | range |
|---|---|---|---|
| 2-class, base | 57.6% | 92.4% | 34.8 pts |
| 2-class, + SE | **84.7%** | 91.0% | **6.3 pts** |
| 4-class, base | 70.8% | 78.5% | 7.7 pts |
| 4-class, + SE | **72.9%** | 75.3% | **2.4 pts** |

**The 4-class arm is the convincing one, precisely because there is no collapse there to
rescue.** All five base seeds were already healthy at ± 3.2%, and the block still cut the
range from 7.7 points to 2.4. That rules out the easy reading ("it only repairs the broken
seed") and supports the stronger one: at this data scale the SE block behaves as a
**regularizer, not as added capacity**. It does not raise the ceiling in either task; it
lowers it slightly while raising the floor a lot.

A third observation points the same way: the SE runs converge earlier. Median epochs before
`EarlyStopping` fires go from 75 to 54 (2-class) and from 112 to 64 (4-class). Fewer epochs
to a more consistent result is the signature of a constrained solution space, not of a model
with more room to fit.

### What it costs the ensemble

Making the seeds behave alike makes them behave alike *as ensemble members too*, and member
disagreement is the precondition for averaging to help at all:

| | mean pairwise disagreement | hard vote |
|---|---|---|
| 2-class | 0.231 → **0.128** | 91.7% → **93.1%** (κ 0.861) |
| 4-class | 0.242 → **0.214** | 77.4% → **75.7%** (κ 0.676) |

In 4-class the trade-off shows up as predicted: less diversity, worse ensemble, 1.7 points
below the base hard vote and 3.5 below the base soft vote (79.2%). In 2-class it does not,
because there the base ensemble was being dragged down by a collapsed member, and removing
the collapse more than pays for the lost diversity. The 93.1% is the best 2-class number in
this repository, but it is 2 test trials away from the base hard vote, which is noise.

Soft voting cannot be compared here: `run_se()` stores `y_pred` but not the softmax
distributions, so only hard voting is available on the SE side. The comparison above is
hard vote against hard vote.

### Reported, not adopted

The SE block is **not** in `train_final.py`, and every headline number in this README is
plain EEGNet. That is deliberate. The mean-accuracy effect is not established, so the only
reason to adopt the block would be that its variance looks better *on `A01E`* — the same
held-out session that Limitations already records as having informed every other design
decision. Adopting it on that basis would be one more instance of exactly the error
documented there. The ablation is reported; the pipeline is unchanged.

Worth stating on the other side of the ledger: unlike the passband, the class count and the
choice of ensemble variant, **the SE block was never selected by looking at `A01E`**. It was
written for a notebook exercise against a within-session split, and the ablation was run
afterwards. Together with the augmentation ablation, this is one of the two comparisons in
the project that test-set selection does not contaminate.

Caveats: five seeds per task, so the Pitman–Morgan test has 3 degrees of freedom and those
p-values are suggestive rather than settled. The test also assumes bivariate normality,
which the 2-class base violates outright, since that distribution is bimodal (collapse
versus no collapse). Single subject A01, as everywhere else here. Per-seed numbers are in
`results_se_ablation.json`; the runs themselves in `runs/se/`.

## What the network actually learned

**The temporal convolution does converge on the physiologically relevant bands — but only
when it has enough trials.** Fed the full 4–40 Hz range, the 4-class model (232 training
trials) puts 7 of its 8 learned filters at peaks between 10.7 and 17.6 Hz, and theta
(4–8 Hz) energy is under 1% in seven of the eight (the remaining filter reaches 1.1%).
Nobody told it which band mattered.

The 2-class model (116 training trials) manages this for only 4 of 8 filters; the other
four degenerate to peaks below 4 Hz — outside the input passband entirely, where there is
no signal to respond to. Halving the training data halves the number of filters that learn
anything. `fig3_temporal_filters.png` shows both models side by side; this contrast is the
most informative interpretability result in the project.

**The spatial convolution does not rediscover C3/C4, and this replicates across all ten
models.** Raw depthwise weights are *filters*, not *patterns*, so reading them as
topographies is invalid — a large weight can serve to cancel noise in a channel rather than
to extract signal from it. They were converted to activation patterns following
[Haufe et al. (2014)](https://doi.org/10.1016/j.neuroimage.2013.10.067) (`A = Cov(X) · w`),
pairing each spatial filter with the covariance of the signal that actually reaches it — the
output of its own temporal filter, not the raw EEG. `spatial_patterns.py` reproduces this.

Taking the twelve C-row and CP-row electrodes as the sensorimotor strip (54.5% of the 22
channels, so that is the level expected if the patterns were spatially indifferent), the
mass that lands there is **49.0% for the 2-class models and 51.2% for the 4-class ones** —
at or below indifference in every one of the ten. C3 and C4 rank **17.5th and 15.7th of 22**
on average. The strongest channels are frontal (Fz, FCz, FC1) or parietal-occipital (POz,
P1, Pz) depending on the seed; only in 4-class do CP3/CP4 and C4 surface in the top three,
and only for some seeds. Claiming this model "rediscovered that C3 and C4 carry the signal"
would be a story the weights do not support — the network reaches 74–92% accuracy without
concentrating on the electrodes the physiology would nominate.

Full per-seed values are in `results_spatial_patterns.json`.

## Within-subject replication across the nine subjects

Everything above this point is subject A01. This section trains the identical pipeline —
same architecture, same augmentation, same batch size, same patience, same 4–40 Hz band —
separately on each of the nine subjects, `A0xT` to `A0xE`. Only the subject changes. It is
replication, not transfer, and it is the denominator every transfer number below is divided
by.

All nine subjects now carry five seeds per task, so mean and SD are reported. While the run
was in progress these tables reported individual values instead: below five seeds the sample
SD carries little, and at two runs it is the gap rescaled by a constant.

<!-- AUTOGEN:within-table -->
| Subject | 2-class, per seed | 4-class, per seed |
|---|---|---|
| A01 *(dev)* | 0.826 [0.576–0.924] | 0.739 [0.708–0.785] |
| A02 | 0.646 [0.514–0.722] | 0.421 [0.392–0.490] |
| A03 | 0.963 [0.944–0.972] | 0.895 [0.882–0.906] |
| A04 | 0.793 [0.729–0.847] | 0.560 [0.542–0.576] |
| A05 | 0.639 [0.535–0.722] | 0.372 [0.309–0.448] |
| A06 | 0.688 [0.625–0.736] | 0.353 [0.257–0.486] |
| A07 | 0.767 [0.674–0.826] | 0.756 [0.719–0.806] |
| A08 | 0.972 [0.958–0.979] | 0.794 [0.760–0.816] |
| A09 | 0.918 [0.910–0.931] | 0.783 [0.767–0.799] |
| **mean of the nine** | **0.801** | **0.630** |

Seeds per cell: 2-class n = 5; 4-class n = 5 (A01 carries five from the single-subject phase). Values are individual runs.
<!-- /AUTOGEN:within-table -->

Two things follow, and the first corrects a claim made earlier in this README.

**A01 is a median subject on this dataset, not a strong one.** The protocol section states
that A01 "is one of the stronger performers", which was asserted without ever running the
other eight. It is not supported: A03, A08 and A09 beat A01 in both tasks, and A04 and A07
beat it in one. A01 sits in the middle of the nine. This does not change any A01 number, but
it removes the reason given for not comparing them against nine-subject means.

**The training instability is a property of the pipeline, not of A01.** The Limitations
section documents the across-seed collapse as an A01 finding with an identified cause — early
stopping on a validation split of 28 trials (2-class) or 56 (4-class). Running the other
eight shows the same failure, landing on different subjects at different seeds:

| | collapsed run | the other four seeds |
|---|---|---|
| A02, 2-class | 0.514, val 0.571, stopped at 32 epochs | 0.653, 0.667, 0.674, 0.722 |
| A05, 2-class | 0.535, val 0.679, stopped at 26 epochs | 0.611, 0.618, 0.708, 0.722 |
| A06, 4-class | 0.257, val 0.232 (below chance), 29 epochs | 0.306, 0.323, 0.396, 0.486 |

The signature is consistent: a collapsed run stops early, and its validation accuracy is as
poor as its test accuracy, so the failure is visible without touching the test set. Gaps
between two seeds of the same subject reach 14 to 17 accuracy points.

This matters beyond replication. The one effect the Squeeze-and-Excitation block was shown
to have is variance reduction — 33.7x and 11.4x on A01, Pitman–Morgan p = 0.016 and 0.030,
on three degrees of freedom. Establishing that the variance it targets is a general property
of the pipeline rather than a quirk of one subject is the precondition for testing that claim
properly. The nine-subject SE arm is not run yet, and it needs five seeds per subject: a
variance-ratio test on two seeds per cell has no power, because with two runs the "variance"
is a single gap.

One hypothesis was recorded at two seeds, before the data that would test it existed, so that
it could not be fitted afterwards: **across-seed dispersion is inversely related to a
subject's mean accuracy.** At five seeds it holds in both tasks.

| | Spearman rho | p | same test at three seeds |
|---|---|---|---|
| 2-class | -0.667 | 0.050 | -0.695, p = 0.038 |
| 4-class | **-0.867** | **0.0025** | -0.183, p = 0.637 |

The 4-class arm is the instructive one. At three seeds the correlation was absent and the
honest reading was that the hypothesis failed there. It had not: at three seeds the
per-subject SD is itself too noisy to correlate with anything, and two more seeds moved it
from rho = -0.183 to rho = -0.867. The test of the hypothesis is also a demonstration of why
this project states seed counts everywhere.

The pattern is that strong subjects are reproducible and weak ones are not. A03, A08 and A09
vary by 1.1 accuracy points or less across five 2-class seeds, while A02 and A05 span 21 and
19 points. A06 produces the widest 4-class spread of any cell, 23 points, at the lowest mean
accuracy of the nine.

The caveats stand: nine points, and the 2-class p = 0.050 sits exactly on the conventional
threshold and is uncorrected. It is reported because it was pre-registered, not because one
marginal p settles anything.

## Cross-subject transfer: the zero-shot floor

The five A01 models per task are applied, unchanged, to every subject's evaluation
session. **Nothing is trained here** — no gradients, no fine-tuning, no adaptation. This
measures the floor that any transfer method has to beat, and it is the denominator for the
freezing experiments that follow. A01 is included as a control: there the model is at home
and must reproduce the within-subject numbers above, which it does exactly, seed by seed.

Chance is not 0.50 and 0.25. The exact one-sided binomial thresholds are **0.576** (2-class,
n=144) and **0.295** (4-class, n=288); corrected for the 16 reported tests (8 test subjects
× 2 tasks), **0.618** and **0.326**. A01 is the development subject and is not one of the 16.

| Target | 2-class members | 2-class soft vote | 4-class members | 4-class soft vote |
|---|---|---|---|---|
| A01 *(dev, control)* | 0.826 ± 0.141 | 0.910 \*\* | 0.739 ± 0.032 | 0.792 \*\* |
| A02 | 0.582 ± 0.026 | 0.632 \*\* | 0.248 ± 0.014 | 0.271 — |
| A03 | 0.757 ± 0.114 | **0.854** \*\* | 0.515 ± 0.056 | **0.552** \*\* |
| A04 | 0.574 ± 0.048 | 0.611 \* | 0.304 ± 0.009 | 0.299 \* |
| A05 | 0.488 ± 0.027 | 0.493 — | 0.258 ± 0.022 | 0.281 — |
| A06 | 0.624 ± 0.049 | 0.667 \*\* | 0.315 ± 0.029 | 0.313 \* |
| A07 | 0.567 ± 0.026 | 0.597 \* | 0.308 ± 0.015 | 0.319 \* |
| A08 | 0.696 ± 0.124 | **0.792** \*\* | 0.400 ± 0.045 | **0.458** \*\* |
| A09 | 0.543 ± 0.047 | 0.569 — | 0.308 ± 0.062 | 0.285 — |

\*\* above chance after Bonferroni over 16 tests; \* above chance uncorrected only; — not
above chance. Five seeds per cell. Full per-seed values, kappas, confidences, pairwise
disagreements and McNemar tests are in `results_zeroshot.json`.

**Transfer is subject-pair specific, not uniformly absent.** Over the eight test subjects,
the soft vote clears the corrected threshold for 4/8 in 2-class and 2/8 in 4-class. A03 and
A08 transfer in both tasks (p < 1e-13). A05 transfers in neither — 0.493 in 2-class is below
the 0.50 chance level outright. Writing "the model does not transfer across subjects" would
be as wrong as writing that it does.

Three of the 4-class cells (A04, A06, A07) sit between the uncorrected and corrected
thresholds. Compared against a naive 0.25 they would each have been called a success.

### The ensemble gain depends entirely on the baseline

This is the part that changes the conclusion, and it goes against what the single-subject
results suggested.

| 2-class, 8 test subjects | gain | subjects improved | Wilcoxon |
|---|---|---|---|
| soft vote vs **mean of the five members** | +0.048 | 8/8 | p = 0.0039 |
| soft vote vs the **member selected on validation** | +0.012 | 4/8 | p = 0.30 |

In 4-class the second row is *negative* (−0.002, 5/8, p = 0.42).

The mean of the members is not a classifier anyone uses; it includes the collapsed seeds.
In practice one model is deployed, chosen on validation data. Against that baseline — here
the member with the highest `val_acc_restored` on `A01T`, seed 1 in both tasks, selected
without ever looking at any evaluation session — the seed ensemble is a coin flip across
subjects. The clearest single case is **A09, 4-class**: the validation-selected member
reaches 0.406, the soft vote 0.285, McNemar p < 0.0001 *against* the ensemble.

Two caveats, in both directions. The p = 0.0039 on the 8/8 row is the smallest value a
one-sided test on eight pairs can return; it means "all eight improved" and nothing
stronger. And the A09 result may be validation selection getting lucky on one subject —
eight subjects cannot distinguish that from validation selecting well in general.

The 4-class ensemble advantage found on A01 (McNemar p = 0.013) **does not replicate**
across the eight unseen subjects. That is the result this phase existed to obtain.

### Normalised by each subject's own ceiling

Absolute transfer accuracy conflates two different things: how well the source model
transports to a person, and how decodable that person is for any model at all. The table
above says A03 and A08 are the subjects that transfer. They are also the two easiest subjects
in the dataset -- their own within-subject models reach 0.963 and 0.972 in 2-class -- so most
of what that reading measured was the subject, not the transport.

Dividing by each subject's own within ceiling separates them. Two quantities are reported.
The **ratio** is the zero-shot member mean over the within mean: single model against single
model, since pitting a five-model vote against one model would inflate the numerator.
**Retained above chance** is the share of the headroom that survives transfer,
`(zero-shot - chance) / (within - chance)`, which is what a transfer method actually has left
to recover.

<!-- AUTOGEN:ratio-table -->
| Target | Within ceiling | Zero-shot | Ratio | Retained above chance |
|---|---|---|---|---|
| **2-class** | | | | |
| A01 *(dev, same person — not transfer)* | 0.826 | 0.826 | 1.00 | — |
| A02 | 0.646 | 0.582 | 0.901 | 56.2% |
| A03 | 0.963 | 0.757 | 0.786 | 55.6% |
| A04 | 0.793 | 0.574 | 0.723 | 25.1% |
| A05 | 0.639 | 0.487 | 0.763 | -9.0% |
| A06 | 0.688 | 0.624 | 0.907 | 65.9% |
| A07 | 0.767 | 0.567 | 0.739 | 25.0% |
| A08 | 0.972 | 0.696 | 0.716 | 41.5% |
| A09 | 0.918 | 0.543 | 0.592 | 10.3% |
| **4-class** | | | | |
| A01 *(dev, same person — not transfer)* | 0.739 | 0.739 | 1.00 | — |
| A02 | 0.421 | 0.248 | 0.589 | -1.2% |
| A03 | 0.895 | 0.515 | 0.576 | 41.1% |
| A04 | 0.560 | 0.303 | 0.542 | 17.2% |
| A05 | 0.372 | 0.258 | 0.694 | 6.8% |
| A06 | 0.353 | 0.315 | 0.892 | 63.1% |
| A07 | 0.756 | 0.308 | 0.408 | 11.5% |
| A08 | 0.794 | 0.400 | 0.504 | 27.6% |
| A09 | 0.783 | 0.308 | 0.393 | 10.8% |
<!-- /AUTOGEN:ratio-table -->

<!-- AUTOGEN:ratio-medians -->
- **2-class:** median ratio **0.75** (range 0.59–0.91); median retained above chance **33.3%** (range -9.0% to 65.9%).
- **4-class:** median ratio **0.56** (range 0.39–0.89); median retained above chance **14.4%** (range -1.2% to 63.1%).
<!-- /AUTOGEN:ratio-medians -->

A flagged row marks a subject whose own within accuracy is not itself above chance, where the
denominator is noise and the percentage is not interpretable.

This reverses the absolute reading. **A06 is the best transfer target in both tasks**,
retaining 65.9% of its headroom in 2-class and 63.1% in 4-class against A03's 55.6% and 41.1%,
despite lower absolute accuracy in every cell. The two subjects the absolute table nominated
as the ones that transfer are not the ones that transfer best. A05's 4-class deficit is
largely its own ceiling (within 0.372 against 0.25 chance), not transport; its 2-class failure
is real, retaining a negative share.

The sharpest cases are **A09 and A07 in 4-class**: within accuracy of 0.783 and 0.756, as good
as A01 or better, yet zero-shot leaves them near chance, retaining 10.8% and 11.5% of the
headroom. These are highly decodable people that this source model does not reach, and they
are where fine-tuning has the most room to act.

**Stability caveat.** The retained-above-chance figure divides by `within - chance`, and that
denominator is itself estimated from the seeds, so it is the least stable number here. Across
one to five seeds the 4-class median held at 14.5%, 14.8%, 14.3%, 14.4%, while the 2-class
median moved 24.0%, 35.4%, 34.2%, 33.3% -- unstable between the first and second seed,
settled from the third on. The plain ratio, whose denominator is the within mean rather than
a difference, was steadier throughout (2-class 0.720, 0.783, 0.771, 0.751; 4-class 0.564,
0.561, 0.557, 0.559). Both are now measured against a five-seed ceiling.

### What this does not establish

- Only **one source subject**. Every number here partly measures similarity to A01, not
  transferability in general. Training on several subjects is a different regime.
- These are not comparable to the ~77% nine-subject means in the literature, which are
  within-subject.
- No average over the nine subjects is reported as a result, because A01 is the development
  subject and is at home.

## Repository layout

```
eeg_fundamentos_e_arquitetura.ipynb   Course notebook: EEG signal, frequency bands,
                                      EEGNet layer by layer, ablations, filter inspection,
                                      and the Squeeze-and-Excitation variant (Exercise 8)
eegnet_motor_imagery.ipynb            First 4-class baseline (no augmentation)
eegnet_mi_improvements.ipynb          Improvement path: 2-class, band tuning, augmentation
train_final.py                        Clean, seeded, reproducible final pipeline
                                      (main runs + the augmentation ablation)
ensemble.py                           Seed ensemble: soft/hard voting, McNemar, diversity
spatial_patterns.py                   Haufe patterns of the depthwise conv (C3/C4 question)
se_ablation.py                        Squeeze-and-Excitation ablation: the Exercise 8 block
                                      under the final protocol, paired seed by seed
make_figures.py                       Regenerates the figures from the saved models
runs/run_*.json                       One JSON per seed, written as each run finishes
runs/ablation/noaug_*.json            Same pipeline with augmentation switched off
runs/se/se_*.json                     Same pipeline with the SE block switched on
results_final.json                    Full per-seed metrics and classification reports
results_ensemble.json                 Ensemble metrics for both tasks
results_ablation.json                 Paired with/without-augmentation comparison
results_se_ablation.json              Paired with/without-SE comparison
results_spatial_patterns.json         Sensorimotor mass and C3/C4 ranks per seed
models/                               Final trained models (~100–115 KB each)
figures/                              Figures produced by make_figures.py
```

## Reproducing

The `.gdf` recordings are not in this repository (~600 MB).

1. Download **BCI Competition IV dataset 2a** from
   [bbci.de/competition/iv](https://www.bbci.de/competition/iv/#dataset2a) (free, requires
   registration) and the **true labels** for the evaluation sessions from
   [bbci.de/competition/iv/results](https://www.bbci.de/competition/iv/results/).
2. Place them as:
   ```
   data/bcic_iv_2a/A01T.gdf  ... A09E.gdf
   data/true_labels/A01E.mat ... A09E.mat
   ```
3. Install and run:
   ```bash
   pip install -r requirements.txt
   python train_final.py     # ~90 min on CPU: 5 seeds per task, plus the ablation
   python ensemble.py
   python spatial_patterns.py
   python se_ablation.py     # ~40 min on CPU: 5 seeds per task with the SE block
   python make_figures.py
   ```

`train_final.py` skips any seed whose `runs/*.json` and `models/*.keras` both exist, so
it is resumable — but that also means a stale artifact is never refreshed. Delete both
files for a seed to force it to retrain. The `.keras` files are tied to the Keras version
that wrote them: models saved before Keras 3 embed `renorm` arguments in
`BatchNormalization` that Keras 3 refuses to deserialize, so after a major upgrade the
seeds must be retrained before `ensemble.py` can load them.

Training is deterministic given the seed, and it is stable across the Keras versions this
was run under. The 2-class models and 4-class seeds 0–2 were trained under Keras 3.13.2;
4-class seeds 3–4 under Keras 3.14.0 (the version pinned in `requirements.txt`), both on
TensorFlow 2.21.0 / MNE 1.12.1 / NumPy 2.4.4. Two seeds retrained from scratch under 3.14.0
reproduced their 3.13.2 accuracy, kappa, confusion matrix and epoch count *exactly*, and all
ten saved models in `models/` reproduce the metrics recorded in their `runs/*.json` when
evaluated under 3.14.0. The SE runs in `runs/se/` span environments the same way: 2-class and
4-class seeds 0–1 were trained first, 4-class seeds 2–4 later under the pinned versions above.
Seeds 0 and 1 of the 4-class arm were retrained from scratch under the pinned environment as a
check and reproduced their accuracy and epoch count exactly (75.35% / 105 epochs and 75.00% /
75 epochs), so the SE comparison is not an artefact of mixing environments. Each `.keras` file records the version that wrote it, in its
`metadata.json`.

TensorFlow runs CPU-only here: GPU support is unavailable on native Windows for TF ≥ 2.11,
and the DirectML plugin does not implement `channels_first` backpropagation, which EEGNet's
depthwise convolution requires.

## Limitations

- **Run-to-run variance is large, and it is the headline caveat.** Across five seeds the
  2-class test accuracy ranges from 57.6% to 92.4%. Four seeds land between 86.8% and
  92.4%; one collapses. Any single-run number from this pipeline — including the 88.9%
  this project originally reported — is a draw from that distribution, not a point
  estimate. This is why the results table above reports the spread.
- The instability has an identifiable cause: early stopping selects on a 28-trial
  validation set (2-class), which is far too small for `val_loss` to be a stable signal.
  The collapsing seed stopped at 41 epochs. Ensembling across seeds is implemented here and
  does address the variance; nested cross-validation, which would also give an unbiased
  estimate of the selection itself, is not. A second thing that addresses it, measured but
  deliberately not adopted, is the SE block: it cuts the across-seed standard deviation by
  33.7× in 2-class and 11.4× in 4-class without changing mean accuracy.
- Validation accuracy is recorded as `val_acc_restored` — the accuracy of the model
  `EarlyStopping` actually restored. The obvious alternative, the peak `val_accuracy` seen
  during training, is optimistic by up to 10 points here (seed 3, 2-class: 67.9% peak vs
  57.1% restored) because the restored epoch is chosen on `val_loss`, not on accuracy. Only
  the restored value is used anywhere a decision is made.
- **`A01E` was consulted at every design decision, so these numbers are an upper bound.**
  The baseline, the passband, the adoption of augmentation, the move to 2 classes and the
  choice of ensemble variant were each evaluated on the same 144 (or 288) test trials, and
  the option that scored higher was kept. The augmentation ablation above was run after the
  fact and is unanimous across ten seeds, which makes that particular decision much harder
  to attribute to test-set noise — but it does not undo the fact that the decision was
  originally made by looking. That is model selection on the test set, one
  decision at a time — the experimenter, not the network, is the optimizer, and the test
  session has in practice been serving as a second validation set. Part of every observed
  gap is noise specific to these trials, and retaining the winner retains that noise, so
  the reported accuracy is optimistic relative to a session that had never been looked at.
  The passband is the clearest case. An 8–30 Hz band has an independent justification —
  mu and beta are the rhythms that desynchronise during motor imagery — but it was tried
  and scored 50.7% on the test set, at chance, so the wider 4–40 Hz band was kept. That
  choice was made *because the number went up*, and it is the test set that told us. No
  correction is applied here; a clean estimate needs a session, or a subject, that has
  never been evaluated.

- **The detailed results sections above are subject A01.** The nine-subject within-subject
  replication is complete at five seeds and has its own section, but the SE ablation, the
  ensemble variants and the spatial-pattern analysis were not repeated on the other eight. The cross-subject section is the one part measured
  on all nine; it establishes a zero-shot floor, not transferability in general, because it
  has a single source subject.
- The cross-subject results use one source subject and no domain alignment beyond the
  per-epoch z-score — which normalises by a single scalar per epoch and therefore corrects
  global gain without aligning the between-channel covariance that the spatial filter
  actually consumes. Euclidean Alignment is the standard baseline here and is not yet run;
  run configs carry an explicit `align` field so that arm is a filter, not a migration.
- 4 s decision windows, so latency is ~4 s. Too slow for responsive game control.

## Next steps

- Sliding-window augmentation with 2 s windows, to halve latency and multiply trial count.
- Confidence-thresholded rejection for online use: the softmax confidence gap between
  correct and incorrect predictions is large enough to act on.
- Online integration over TCP with an OpenBCI headset (separate repository).

## References

- Lawhern et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces.* J. Neural Eng. 15(5). [doi](https://doi.org/10.1088/1741-2552/aace8c)
- Haufe et al. (2014). *On the interpretation of weight vectors of linear models in
  multivariate neuroimaging.* NeuroImage 87. [doi](https://doi.org/10.1016/j.neuroimage.2013.10.067)
- Brunner et al. (2008). *BCI Competition 2008 – Graz data set A.*
- Pfurtscheller & Neuper (2001). *Motor imagery and direct brain–computer communication.*
