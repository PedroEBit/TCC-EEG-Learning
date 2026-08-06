<!--
Versao em ingles, revisada contra o codigo: todo numero aqui sai de
results_final.json ou da analise de pesos em make_figures.py.
-->

My EEG classifier had learned to always guess "right hand."

144 training trials, two classes, and the model landed on 50.7% accuracy — Cohen's κ of 0.01. It wasn't confused. It had found the laziest possible strategy: predict one class every time, and be right half the time. It answered "right" on 133 of 144 trials.

The task: classify motor imagery from EEG. When you imagine moving your left hand, the mu rhythm (8–13 Hz) desynchronizes over the contralateral motor cortex. That modulation is real, and it is buried in noise. 144 trials to find it, against a network with 4,000 parameters — training accuracy climbed to 84% while validation sat at 45%.

**What actually fixed it**

Not a bigger network. Data augmentation via Gaussian noise injection — additive noise scaled to each epoch's own standard deviation at σ = 0.1, simulating the trial-to-trial variability of background EEG. The ERD/ERS signal is a slow modulation of mu/beta power, so it survives additive noise that destroys nothing but the background.

Five noisy copies per trial. Augmentation touches the training split only.

On leakage, because it is the first thing worth asking: there is no random train/test split here. BCI Competition IV-2a ships each subject as two sessions recorded on different days — I train on session T and test on session E. Augmentation is structurally incapable of crossing that boundary.

Then I ran it five times with different random seeds, and the result fell apart.

2-class accuracy across five seeds: 88.2%, 88.2%, 92.4%, **57.6%**, 86.8%. Same data, same architecture, same hyperparameters. One run in five simply fails to converge. Any single number from this pipeline — including the 88.9% I had been quoting — is a draw from that distribution, not a measurement.

**What I did about it**

Stopped picking a model. Averaged the softmax outputs of all the seeds instead. Crucially, each seed also draws its own train/validation partition, so each member has seen a different 116 of the 144 trials and they make substantially different errors — mean pairwise disagreement is 0.23. That decorrelation is the whole reason averaging works.

Results on the held-out session, within-subject, subject A01:
· 2-class (left vs. right hand): **91.0% accuracy, κ = 0.819**
· 4-class (left, right, feet, tongue): **78.5% accuracy, κ = 0.713**

The 4-class ensemble beats *every* individual member — 78.5% against a best single seed of 75.7% — and a McNemar test against the median member gives p = 0.023. The 2-class ensemble lands at 91.0%, above the median member, but there p = 0.29: the improvement points the right way and 144 test trials cannot prove it. I am reporting both p-values, including the one that fails.

No member was ever selected or dropped using the test set. The collapsed seed is still in there, diluted rather than deleted.

I report κ because it is the BCI Competition standard — it discounts agreement by chance. And I report the subject, because these are within-subject results on one of the dataset's stronger subjects. They are not comparable to the nine-subject averages usually quoted for this benchmark.

**The bug I shipped and had to fix**

Keras' `validation_split=0.2` reserves the last 20% of the array *without shuffling*. My augmented array was `[originals, copy1, ..., copy5]` — so the validation set was noisy duplicates of trials already in training. Validation accuracy read 1.000. The test numbers were never affected, since the test set is a physically separate session, but early stopping was choosing epochs on a meaningless signal.

Splitting validation off *before* augmenting costs 20% of the training data, and it cost accuracy too: single-run 4-class fell from 80.2% to 73.2%. That number had been propped up by the broken early stopping, which let the model train far past where an honest validation signal would have halted it.

Watching a number get worse because you fixed something is the most useful signal you can get. It is also the reason the ensemble was worth building rather than optional.

**Why EEGNet's layers map onto neurophysiology**

The architecture (Lawhern et al., 2018) is not a generic CNN, and the weights show it.

Its temporal convolution learns frequency filters. I fed it the full 4–40 Hz band and went to look at what it converged on. In the 4-class model, 7 of its 8 learned filters peak between 10.7 and 17.6 Hz — the mu/beta range — and theta energy is under 1% in every single one. Nobody told it which band mattered.

In the 2-class model, trained on half as many trials, only 4 of 8 do this. The other four degenerate to peaks below 4 Hz, outside the input passband entirely, where there is no signal to respond to. Halve the data, halve the filters that learn anything. That contrast is the clearest picture I have of what 144 trials actually costs you.

Its depthwise spatial convolution learns channel combinations — and here the textbook story did not survive contact with the weights. Raw depthwise weights are *filters*, not *patterns*, so reading them as topographies is invalid; you have to convert them to activation patterns first (Haufe et al., 2014). Do that, and sensorimotor channels carry 44–47% of the pattern mass — *below* the 50% you would get by chance. C3 and C4 rank around 13th of 22. What lateralization exists sits on CP3/CP4, a few centimetres posterior.

I had written that the network "rediscovers that C3 and C4 carry the discriminative signal." It does not. I checked all three trained models and the negative result replicated in every one.

Walking that claim back is the part I would not have known how to do six months ago.

**The finding I did not expect**

Softmax confidence separates by correctness: 87.8% mean confidence on correct predictions vs. 71.0% on incorrect ones. That gap is not a curiosity — it is a usable rejection threshold. In an online BCI, refusing to act on a low-confidence prediction beats acting on a wrong one. For game control specifically, precision matters more than recall: moving the wrong way is worse than not moving.

Next: sliding-window augmentation to cut the 4s decision window in half, and online integration over TCP to control a game in real time.

A few months ago I did not know what a convolution was. The accuracy number is not what I am proud of — it is that I can now tell which of my claims the weights actually support, and which ones I had to walk back.

Dataset: BCI Competition IV-2a (Graz) · Architecture: EEGNet · Stack: Python, MNE, TensorFlow/Keras
Code: github.com/PedroEBit/TCC-EEG-Learning

If you have worked on motor-imagery decoding — what moved the needle most for you on small trial counts?

#BCI #EEG #DeepLearning #MachineLearning #Neuroscience
