# What this repository supports, and what it does not

Written 2026-10-02 for whoever drafts the paper, human or otherwise. Every number below
comes from a JSON file in this repository and can be regenerated. The README is the long
form; this file exists so that a claim is not made here that the data does not carry.

**If a statement is not in the "supported" list, do not write it.**

---

## Experimental setup

BCI Competition IV dataset 2a. Nine subjects, 22 EEG channels, two sessions recorded on
different days. EEGNet, 4-40 Hz IIR bandpass, epochs 0-4 s after cue, z-scored per epoch,
Gaussian-noise augmentation (sigma = 0.1 per epoch, five copies) on the training split only.
Training is `A0xT`, testing is `A0xE` — the official competition split, so no random
train/test partition exists and augmentation cannot leak across it.

**Subject split, declared before use:** development A01, test A02-A09. A01 was consulted at
every design decision in the single-subject phase and is spent; the other eight were never
used to select anything.

**Seeds: five per subject per task, all nine subjects.** The within-subject replication
finished on 2026-10-02 at 19:49. Every number below is at five seeds, and the result JSONs,
the README tables and this file are all generated from the same runs.

---

## Supported

### 1. Within-subject replication across the nine subjects

Mean accuracy over the nine: **0.801** (2-class) and **0.630** (4-class). Per-subject values
in `results_within9.json`. Range 0.639-0.972 and 0.354-0.895.

### 2. A01 is a median subject, not a strong one

A03, A08 and A09 beat A01 in both tasks; A04 and A07 in one. An earlier version of this
README asserted the opposite without having run the other eight. The correction is in the
README's protocol section.

### 3. The across-seed collapse is a property of the pipeline, not of A01

Collapsed runs appear in A02 (2-class: 0.514 against 0.653, 0.667, 0.674, 0.722 on the other
four seeds), A05 (2-class: 0.535 against 0.611, 0.618, 0.708, 0.722) and A06 (4-class: 0.257
with validation 0.232, below chance, against 0.306, 0.323, 0.396, 0.486).

The signature is consistent and diagnostic: a collapsed run stops early (26-41 epochs against
a 150-300 cap) and its validation accuracy is as poor as its test accuracy, so **the failure
is detectable without touching the test set**.

There is no "unlucky seed". Seed 3 is the worst of five for A01 in 2-class (0.576) and the
best of five for A02 in both tasks. `stratified_split` derives the train/validation partition
from the seed, so a seed that produces a poor split for one subject's trials produces a good
one for another's. The instability cannot be fixed by avoiding a seed; it needs a structural
remedy (larger validation split, ensembling, or the SE block).

### 4. Zero-shot cross-subject transfer, normalised

The five A01 models per task applied unchanged to every subject's evaluation session. No
training, no adaptation. Against exact one-sided binomial thresholds corrected over the 16
reported tests, the soft vote clears chance for **4/8** test subjects in 2-class and **2/8**
in 4-class.

Normalised by each subject's own within ceiling — which separates how well the source
transports from how decodable the target is:

| | median ratio | median retained above chance |
|---|---|---|
| 2-class | **0.751** (range 0.592-0.907) | 33.3% (range -9.0% to 65.9%) |
| 4-class | **0.559** (range 0.393-0.892) | **14.4%** (range -1.2% to 63.1%) |

The normalisation reverses the absolute reading: **A06 is the best transfer target in both
tasks**, retaining 65.9% of its headroom in 2-class and 63.1% in 4-class against 55.6% and
41.1% for A03, despite lower absolute accuracy in every cell. A03 and A08 look like the
subjects that transfer only because they are the two easiest subjects in the dataset.

A09 and A07 in 4-class are the sharpest cases: within accuracy 0.783 and 0.756, as good as
A01 or better, yet they retain 10.8% and 11.5% of their headroom under transfer.

### 5. The seed ensemble does not survive the right baseline

Against the **mean of the five members**, the soft vote wins 8/8 test subjects in 2-class
(Wilcoxon p = 0.0039). Against the **member selected on validation data** — highest
`val_acc_restored` on `A01T`, seed 1 in both tasks, chosen without looking at any evaluation
session — it wins **4/8** (p = 0.30), and in 4-class the mean gain is negative.

The mean of the members is not a classifier anyone deploys; it includes the collapsed seeds.
The 4-class ensemble advantage reported for A01 (McNemar p = 0.013) **does not replicate**
across the eight unseen subjects.

### 6. Pre-registered, and it half-held

Recorded at two seeds, before the data that would test it existed: across-seed dispersion is
inversely related to a subject's mean accuracy. At five seeds it holds in both tasks:
Spearman **rho = -0.667, p = 0.050** in 2-class and **rho = -0.867, p = 0.0025** in 4-class.

The 4-class arm is worth reporting in full. At three seeds it read rho = -0.183, p = 0.637,
and the honest conclusion at that point was that the hypothesis failed in that task. It had
not: with three seeds the per-subject SD is itself too noisy to correlate with anything. This
is a result about seed counts as much as about dispersion.

Caveats: nine points, and the 2-class p = 0.050 sits exactly on the conventional threshold
and is uncorrected. It is reported because it was pre-registered.

---

## Not supported — do not write these

- **Any average over the nine subjects as a transfer result.** A01 is the development subject
  and is at home; it is not a transfer case.
- **"The ensemble improves cross-subject transfer."** Against the realistic baseline there is
  no evidence.
- **"The model does not transfer across subjects."** Four of eight clear corrected chance in
  2-class. Equally, **"the model transfers"** is wrong: four of eight do not.
- **Comparison against the ~77% nine-subject figures usually quoted for this dataset.** Those
  are within-subject. The transfer numbers here are zero-shot from one source subject.
- **"A05 is a subject that cannot be decoded."** A05 reaches 0.639 in 2-class on its own
  model. Its 2-class transfer failure is transport, not the subject.
- **Any claim about the Squeeze-and-Excitation block across the nine subjects.** That arm has
  not been run. The A01-only SE result (variance reduced 33.7x and 11.4x, equivalently
  5.80x and 3.38x in standard deviation, Pitman-Morgan
  p = 0.016 and 0.030) rests on three degrees of freedom and is reported as such.
- **Any claim about layer freezing, fine-tuning or calibration budget.** Not run. This
  repository measures the zero-shot floor those experiments would have to beat.
- **Any claim about Euclidean Alignment or domain adaptation.** Not run.

---

## Limitations the paper must state

1. **One source subject.** Every transfer number partly measures similarity to A01.
2. **Five seeds per subject per task.** Enough to estimate dispersion, not enough to pin it
   down: the 2-class dispersion correlation lands at p = 0.050.
3. **No domain alignment.** The per-epoch z-score divides by a single scalar per epoch, so it
   corrects global gain and does not align the between-channel covariance that the spatial
   filter consumes. Euclidean Alignment is the standard baseline here and is not run; the run
   schema carries an explicit `align` field so that arm is a filter rather than a migration.
4. **The 2-class retained-above-chance figure was unstable until the third seed** (24.0%,
   35.4%, 34.2%, 33.3% across one to five seeds). The 4-class figure held throughout (14.5%,
   14.8%, 14.3%, 14.4%). Its denominator is a difference estimated from the seeds, which is
   why it moves most.
5. **The A01 single-subject numbers are an upper bound.** `A01E` was consulted at every design
   decision in that phase — see the README's Limitations section, which documents this in
   detail and does not correct for it.
6. **Spatial filters do not recover C3/C4.** Haufe patterns put 49.0% and 51.2% of their mass
   on the sensorimotor strip against 54.5% expected under spatial indifference. The network
   reaches its accuracy without concentrating on the electrodes the physiology nominates.

---

## Regenerating every number

```
.venv\Scripts\python.exe within9.py --collect        # results_within9.json
.venv\Scripts\python.exe zeroshot.py --collect       # results_zeroshot.json
.venv\Scripts\python.exe transfer_ratio.py           # results_transfer_ratio.json
.venv\Scripts\python.exe make_readme_sections.py     # rewrites the README tables
```

`paths.py` run directly lists every run on disk. Every run JSON carries a complete `config`
block with the git commit and library versions; filter on the config, never on the filename.
