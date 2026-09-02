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
Subject A01 is one of the stronger performers in this dataset, so these numbers should
not be compared directly against the nine-subject means usually quoted in the literature.

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
help at all.

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

## Repository layout

```
eeg_fundamentos_e_arquitetura.ipynb   Course notebook: EEG signal, frequency bands,
                                      EEGNet layer by layer, ablations, filter inspection
eegnet_motor_imagery.ipynb            First 4-class baseline (no augmentation)
eegnet_mi_improvements.ipynb          Improvement path: 2-class, band tuning, augmentation
train_final.py                        Clean, seeded, reproducible final pipeline
                                      (main runs + the augmentation ablation)
ensemble.py                           Seed ensemble: soft/hard voting, McNemar, diversity
spatial_patterns.py                   Haufe patterns of the depthwise conv (C3/C4 question)
make_figures.py                       Regenerates the figures from the saved models
runs/run_*.json                       One JSON per seed, written as each run finishes
runs/ablation/noaug_*.json            Same pipeline with augmentation switched off
results_final.json                    Full per-seed metrics and classification reports
results_ensemble.json                 Ensemble metrics for both tasks
results_ablation.json                 Paired with/without-augmentation comparison
results_spatial_patterns.json         Sensorimotor mass and C3/C4 ranks per seed
models/                               Final trained models (~100–115 KB each)
linkedin_post/                        Figures and post draft
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
evaluated under 3.14.0. Each `.keras` file records the version that wrote it, in its
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
  estimate of the selection itself, is not.
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

- Single subject (A01). Nothing here says the model transfers across subjects — on this
  dataset it generally does not without adaptation.
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
