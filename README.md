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
| Train (`A01T`), 2-class | 144 → 115 train / 29 validation → 690 after augmentation |
| Train (`A01T`), 4-class | 288 → 232 train / 56 validation → 1392 after augmentation |
| Test (`A01E`), 2-class | 144, untouched |
| Test (`A01E`), 4-class | 288, untouched |

Preprocessing: 22 EEG channels, IIR bandpass 4–40 Hz, epochs 0–4 s after cue, z-scored
per epoch (per-epoch statistics only, so no dataset-level statistic crosses the split).

## Results

Test set is the held-out `A01E` session. Every configuration was run with multiple random
seeds, because a single run is not a result — see Limitations.

| | seeds | accuracy (mean ± sd) | median | range | Cohen's κ (mean ± sd) |
|---|---|---|---|---|---|
| 2-class (left vs. right hand) | 5 | 82.6% ± 14.1% | **88.2%** | 57.6 – 92.4% | 0.653 ± 0.283 |
| 4-class (left, right, feet, tongue) | 3 | **73.2% ± 2.4%** | 72.9% | 70.8 – 75.7% | 0.642 ± 0.033 |

Chance is 50% and 25% respectively. Per-seed numbers, confusion matrices and full
classification reports are in `results_final.json`.

The **4-class model is far more stable than the 2-class one** (± 2.4% vs ± 14.1%), which
is the opposite of what task difficulty alone would predict. It has twice the training
data — 232 trials against 115 — and, just as importantly, a 56-trial validation set
instead of 29, so early stopping has a usable signal. Data quantity, not task difficulty,
is what governs reliability at this scale.

### Seed ensemble

Because run-to-run variance is the dominant error source, the primary result is an
ensemble: every seed's model is kept, and their softmax distributions are averaged
(soft voting). No member is ever selected or discarded using the test set.

| | members | best single member* | ensemble accuracy | ensemble κ | McNemar vs median member |
|---|---|---|---|---|---|
| 2-class | 5 | 92.4% | **91.0%** | 0.819 | p = 0.29 |
| 4-class | 3 | 75.7% | **78.5%** | 0.713 | **p = 0.023** |

\* not selectable in advance — you only know which member was best after seeing the test set.

The 4-class ensemble **beats every individual member** (78.5% against a best of 75.7%),
and the improvement over the median member is statistically significant. The 2-class
ensemble lands between the median member (88.2%) and the best (92.4%); its improvement is
in the expected direction but 144 test trials are not enough to establish it (p = 0.29).

Mean pairwise disagreement between members is 0.231 (2-class) and 0.259 (4-class) — the
members make substantially different errors, which is the precondition for averaging to
help at all.

A secondary variant discards members more than one standard deviation below the mean
*validation* accuracy — a rule computable without the test set. It raises 2-class to
92.4% (dropping the collapsed seed) but *lowers* 4-class to 77.8%, because with only three
members a standard-deviation rule is unstable. The all-members ensemble is therefore
reported as the primary result, since it requires no member-selection rule at all.

Full ensemble metrics are in `results_ensemble.json`; `explicando_ensemble.ipynb` derives
the reasoning from scratch.

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
figure does not: it settles around 73%. The earlier 80.2% benefited from broken early
stopping, which let that model train far longer than a clean validation signal would have
allowed. The notebook numbers should be read as superseded by the table above.

## Data augmentation

Additive Gaussian noise, σ = 0.1 scaled to each epoch's own standard deviation, five
copies per trial. It simulates the trial-to-trial variability of background EEG while
leaving the ERD/ERS modulation — a slow change in mu/beta power — intact.

```python
noise = rng.standard_normal(X.shape) * (0.1 * X.std(axis=(1, 2), keepdims=True))
```

**The validation split happens before augmentation.** This is the one thing worth copying
from this repo. Keras' `validation_split=0.2` reserves the *last 20% of the array without
shuffling*; on an array built as `[originals, copy1, ..., copy5]` those last rows are noisy
duplicates of trials already in training. The symptom is unmistakable — validation accuracy
pinned at 1.000 while test accuracy sits far below — and it makes `EarlyStopping` select
epochs on a meaningless signal. It never contaminated the test set here (that is a separate
session file), but it did have to be fixed. See `split_then_augment()` in `train_final.py`.

## What the network actually learned

**The temporal convolution does converge on the physiologically relevant bands — but only
when it has enough trials.** Fed the full 4–40 Hz range, the 4-class model (232 training
trials) puts 7 of its 8 learned filters at peaks between 10.7 and 17.6 Hz, and theta
(4–8 Hz) energy is under 1% in *every one of them*. Nobody told it which band mattered.

The 2-class model (115 training trials) manages this for only 4 of 8 filters; the other
four degenerate to peaks below 4 Hz — outside the input passband entirely, where there is
no signal to respond to. Halving the training data halves the number of filters that learn
anything. `fig3_temporal_filters.png` shows both models side by side; this contrast is the
most informative interpretability result in the project.

**The spatial convolution does not rediscover C3/C4, and this replicates.** Raw depthwise
weights are *filters*, not *patterns*, so reading them as topographies is invalid — they
were converted to activation patterns following
[Haufe et al. (2014)](https://doi.org/10.1016/j.neuroimage.2013.10.067) (`A = Cov(X) · w`).
Across all three trained models the mass of those patterns over sensorimotor channels sits
at 44–47%, i.e. *below* the 50% expected by chance, and C3 and C4 rank 12th–14th of 22 on
average. Where laterality does appear it is centred on **CP3/CP4**, a few centimetres
posterior, over somatosensory cortex. Claiming this model "rediscovered that C3 and C4
carry the signal" would be a story the weights do not support.

## Repository layout

```
eeg_fundamentos_e_arquitetura.ipynb   Course notebook: EEG signal, frequency bands,
                                      EEGNet layer by layer, ablations, filter inspection
eegnet_motor_imagery.ipynb            First 4-class baseline (no augmentation)
eegnet_mi_improvements.ipynb          Improvement path: 2-class, band tuning, augmentation
train_final.py                        Clean, seeded, reproducible final pipeline
make_figures.py                       Regenerates the figures from the saved models
results_final.json                    Full per-seed metrics and classification reports
models/                               Final trained models (~115 KB each)
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
   python train_final.py     # ~40 min on CPU, 5 seeds (2-class) + 3 seeds (4-class)
   python make_figures.py
   ```

TensorFlow runs CPU-only here: GPU support is unavailable on native Windows for TF ≥ 2.11,
and the DirectML plugin does not implement `channels_first` backpropagation, which EEGNet's
depthwise convolution requires.

## Limitations

- **Run-to-run variance is large, and it is the headline caveat.** Across five seeds the
  2-class test accuracy ranges from 57.6% to 92.4%. Four seeds land between 86.8% and
  92.4%; one collapses. Any single-run number from this pipeline — including the 88.9%
  this project originally reported — is a draw from that distribution, not a point
  estimate. This is why the results table below reports the spread.
- The instability has an identifiable cause: early stopping selects on a 29-trial
  validation set (2-class), which is far too small for `val_loss` to be a stable signal.
  The collapsing seed stopped at 41 epochs. Nested cross-validation, or ensembling across
  seeds, is the correct fix and is not implemented here.
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
