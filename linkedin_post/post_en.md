<!--
Post de feed do LinkedIn (limite de 3.000 caracteres).
Todo numero aqui sai de results_final.json, results_ensemble.json,
results_ablation.json ou results_spatial_patterns.json.
Ajuste "At the start of this year" para o mes real em que voce comecou.
-->

At the start of this year I didn't know what a convolution was. Today I can tell you which of my own results I had to throw out — and that turned out to be the real skill.

My thesis: decode motor imagery from EEG. You imagine moving your left hand, the mu rhythm (8–13 Hz) desynchronizes over the opposite motor cortex, and a 4,000-parameter CNN (EEGNet) tries to read that from 144 trials.

I hit 88.9% and was thrilled. Then I started checking my own work.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝟭 — my validation set was fake.
Keras' validation_split takes the last 20% of the array without shuffling. Mine was [originals, noisy copy 1 … copy 5]. So validation was noisy duplicates of training trials. Validation accuracy read 1.000. Early stopping had been picking epochs off noise.
Fixed it. 4-class dropped from 80.2% to 74%.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝟮 — one run is not a result.
Five seeds, 2-class: 88.2, 88.2, 92.4, 𝟱𝟳.𝟲, 86.8. Same data, same code. My 88.9% was a lucky draw.
Fix: stop picking a model. Average all five seeds' softmax — the collapsed one included, diluted rather than deleted. Each seed also draws its own train/val split, so members see a different 116 of 144 trials and make different errors — that decorrelation is why averaging works.
→ 2-class 91.0% (κ 0.82) · 4-class 79.2% (κ 0.72), McNemar p = 0.013 vs the median seed.
The ensemble doesn't repair the bad seed. It removes my exposure to which seed I drew.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝟯 — was my augmentation doing anything?
I'd never tested it under the corrected protocol. Ablation, 5 seeds per task: +17.4 and +12.7 points, better in 5 of 5 seeds. But not as a uniform lift — without it, three of five 2-class seeds collapse to chance. It buys reliability, not accuracy.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝟰 — the good story was false.
The temporal filters did find the physiology: 7 of 8 peak between 10.7 and 17.6 Hz, unprompted. So I wrote that the spatial layer "rediscovers C3/C4."
It doesn't. Depthwise weights are filters, not patterns — you must convert them (Haufe et al., 2014) before reading them as topographies. Done properly, sensorimotor mass sits at 49–51% against the 54.5% you'd get from spatial indifference. C3 and C4 rank ~16th of 22. Negative result, replicated in all ten models.
I deleted the claim.

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝟱 — the one I couldn't fix.
The test session informed every design decision: the passband, the augmentation, the move to 2 classes. That's model selection on the test set, one decision at a time. My numbers are an upper bound, and the README says so in full.

What I'd want an employer to take from this: the accuracy isn't the point. Anyone can report a number. I can tell you which of mine survive scrutiny — and I found every one of these myself.

Open to ML / neurotech roles, remote or relocating.

Code: github.com/PedroEBit/TCC-EEG-Learning
BCI Competition IV-2a · EEGNet · Python, MNE, TensorFlow

#BCI #EEG #DeepLearning #MachineLearning #Neurotech
