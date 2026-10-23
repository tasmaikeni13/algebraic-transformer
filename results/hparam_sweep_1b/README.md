# 1B hyperparameter sweep

Completed 2026-10-23 11:17 IST. Six arms trained to a 5B-token target per arm, using seed 42, a 32,768-token LLaMA-style BPE vocabulary, and a 10M-token held-out FineWeb-Edu validation set. The run used 16 TPU devices across four workers.

| Architecture | Learning rate | Mean logged train loss | Final eval NLL | Final eval PPL | Mean train tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 6e-4 | 2.912033 | 2.556442 | 12.889873 | 221,749 |
| Algebraic | 6e-4 | 2.868072 | 2.519763 | 12.425653 | 210,262 |
| Standard | 1.2e-3 | 2.911943 | 2.527480 | 12.521905 | 221,750 |
| Algebraic | 1.2e-3 | 2.832356 | 2.473938 | 11.869097 | 210,252 |
| Standard | 2.4e-3 | 3.061526 | 2.611553 | 13.620192 | 221,750 |
| Algebraic | 2.4e-3 | 2.860457 | 2.434470 | 11.409764 | 210,256 |

The selected learning rates by final validation NLL are 1.2e-3 for Standard (PPL 12.521905) and 2.4e-3 for Algebraic (PPL 11.409764). The selected Algebraic result has 8.88% lower PPL than the selected Standard result on this validation set.

Mean train loss is the arithmetic mean over the 191 periodic training-loss log entries per arm. Learning rates were selected using this validation set, with one seed per arm; this sweep is evidence at this model/data scale and does not establish that the difference persists at larger scales.
