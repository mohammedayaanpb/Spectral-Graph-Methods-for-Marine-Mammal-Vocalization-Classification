# Methods and evaluation

The project analyzes four-family classification, classification of 17 Delphinidae
species, and acoustic clustering within Spinner Dolphin recordings.
Dataset provenance is documented in [DATA.md](DATA.md).

## Audio representations

Recordings are resampled to 22,050 Hz and segmented into non-overlapping
two-second windows. Short recordings are symmetrically zero-padded. A final
partial window is retained and padded when it is at least one second long.
Each window is filtered with a fourth-order, 100 Hz–10 kHz Butterworth bandpass.

| Representation | Features |
| --- | --- |
| Level 1: 32 dimensions | Means of 13 MFCCs, 13 delta-MFCCs, and six spectral/energy statistics |
| Level 2: 51 dimensions | Level 1 features, seven spectral-contrast bands, and 12 chroma bins |
| Level 2 alternative: 51 dimensions | Chroma replaced by standard deviations of six MFCCs and six delta-MFCCs |
| Level 3: 46 dimensions | MFCC, spectral, energy, temporal-entropy, envelope-kurtosis, and autocorrelation features |
| Neural-network inputs | Log-mel spectrograms with 128 mel bands |

## Graph methods

Standardized acoustic features define symmetric nearest-neighbor graphs with
Gaussian weights, using either a global bandwidth or a local self-tuning bandwidth.
Utilities implement harmonic label propagation, Laplacian regularization,
diffusion maps, and spectral clustering with normalized-Laplacian eigenvectors
followed by k-means. Label propagation uses the unnormalized Laplacian.
Level 2 includes neighbor-count/bandwidth sweeps and a connectivity correction
that joins disconnected components to an anchor component with weak edges.

## Classification protocols

The graph experiments are **transductive**: feature standardization and graph
construction use all windows in the chosen subset. V2 selects labeled clips
within each class across the dataset's original train and test partitions.
Every window belonging to a selected clip receives its label; the remaining
windows form the evaluation set. Reported graph summaries average accuracy and
macro F1 over 20 label selections, with standard deviations across selections.
Level 1 uses a 50% clip-label budget; Level 2 evaluates 30%, 50%, and 75% budgets.

The V2 ResNet-18 experiments use the source dataset's held-out test partition.
Subset runs train on the intersection of the original training partition and a
saved graph label mask; full-label runs train on the entire training partition.
Training uses 50 epochs, Adam at learning rate 0.001, batch size 32, and seed 0.
Each saved CNN configuration represents one run. Graph and CNN evaluation cohorts
and label budgets differ, so their values do not establish a controlled ranking.

Classification metrics are computed over **windows**. Accuracy therefore gives
long recordings more weight, while macro F1 averages the window-based class F1
scores. Clip-level label selection prevents a recording from straddling the
graph's labeled/unlabeled boundary; it does not make windows independent samples.

## Acoustic clustering and V2 changes

V2 caps Spinner Dolphin contributions at three windows per recording, yielding
122 windows from 114 clips. Spectral clustering is evaluated with silhouette
scores in standardized feature space. The autoencoder baseline clusters its
learned bottleneck representation and computes silhouettes in that space.

Acoustic proxy labels are formed using k-means on spectral entropy, dominant-bin
variance, and peak-to-average power ratio (PAPR), or by dividing PAPR into thirds.
Adjusted Rand index and normalized mutual information measure agreement between
these proxies and the spectral clusters. These are acoustic agreement measures,
not expert annotations of biological call types. The proxies use the same audio;
different features and low pairwise correlations do not establish independence.

Relative to V1, V2 implements clip-level label selection, common 50-epoch CNN
settings, capped Level 3 recording contributions, additional acoustic proxies,
graph connectivity correction, and a Level 2 feature ablation.
