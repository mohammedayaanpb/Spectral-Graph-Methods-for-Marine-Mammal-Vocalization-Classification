# Data provenance

Source: [confit/wmms-parquet on Hugging Face](https://huggingface.co/datasets/confit/wmms-parquet),
a Parquet distribution of the Watkins Marine Mammal Sound Database.
Credit: Watkins Marine Mammal Sound Database, Woods Hole Oceanographic
Institution and the New Bedford Whaling Museum.
The dataset card links the [original archive](https://archive.org/details/watkins_202104).

The card describes personal or academic, noncommercial use. This repository's
MIT license applies to original code; it does not grant rights to third-party
recordings or associated source metadata. Consult the upstream terms for your use.

The inventory contains 1,697 clips and 32 species. The 1,357/340 train/test split
comes from the Hugging Face packaging, not an official biological benchmark split.
`wmmd_metadata.csv` adds project taxonomy and audio-header measurements; its `index`
is within each `split`. Audio filenames in its `path` column are source identifiers.

The historical local Arrow cache records dataset revision
`a90a38e0006991f0c6f6d4e05261949a1da7f14e`. The download notebook pins that revision
when creating a new cache. Existing caches are reused as-is. Row order matters:
preprocessing derives numeric clip IDs from train rows followed by test rows.

## Local files

- `Mss/data/wmms_parquet_local/`: saved Hugging Face DatasetDict; keep locally.
- `Level{1,2,3}/Data/`: original features and mel spectrograms used by V1 and V2.
- `v2/Level*/Data/`: revised features, masks, capped pools, and derived diagnostics.
- `v2/Level*/runs/`: additional local run outputs; only curated run2 evidence is
  included in the public snapshot.

Audio is downloaded separately. Arrow/Parquet files, feature arrays, and model
checkpoints are stored locally; curated figures and result JSON accompany the code.

## Interpretation

Taxonomy comes from the project's manually specified mapping. These are historical
labels, not a current taxonomic authority. The dataset has substantial imbalance;
window counts vary greatly by recording length. Avoid treating windows from one
recording as independent samples or inferring biological call types from clustering
proxies alone. See [methods and evaluation](METHODS.md).
