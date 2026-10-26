# Input data and cache schemas

## Included aggregates

`data/aggregates/all_85_runs_percent.csv`: subject, group, seed, space, top1, top5. Accuracies are percentages. 85 distinct subject/group/seed units × 3 reference spaces = 255 rows. Mean and sample SD use five paired independent initializations per final condition. Other E4/E5 CSV `aux_top1`, `VT_top1`, etc are proportions, not percentages. Reused cells in different comparison CSVs are not additional models. `paired_intervals.json` contains recorded percentage-point effects and crossed-bootstrap percentile intervals; those intervals are not recomputed here. `legacy_X4_summary.csv` is an archived protocol.

## Required external cache

For each replication `validation`/`final` phase, or development `fold1` phase:

```text
<configured_cache>/<phase>/
  data.npz
  brain_pca.npz
  fit_unique_z.npy
```

`data.npz` uses `train_brain`, `train_image`, `train_text`, `train_label`, `eval_brain`, `eval_image`, `eval_text`, `eval_label`, `stimulus_id`. Brain and semantic rows must already match by stimulus identity. Training labels are category IDs. The preparation script may retain additional bookkeeping arrays; preserve them.

`brain_pca.npz`: `components` (d_b,M), `mean` (M), `legacy_to_unique` (M integer entries in [0,D)), `coordinates` (D,3). Archived preparation also stores `raw_mean`, `raw_std`, `fit_rows`, retained variance and voxel identity fields. `fit_unique_z.npy` is fitted standardized responses (N_fit,D). Graph nodes correspond to unique measured voxels; coordinates are native numerical coordinates with unverified world registration/physical units.

## Preparation inputs

The archived sub04 preparation uses seven ROI flags in `GenericObjectDecoding-v2/Subject4.h5`; GOD-Wiki train/test MAT files and frozen category text features; training images in the inherited dataset layout; test JPEG images; and `repvgg_model.safetensors` in the configured cache root. Inspect `src/sivr/replication/prepare.py` for exact inherited filenames. Acquisition and licensing of these inputs are external steps, not automatic downloads in this repository.

Class selection fits on permitted 120/30 class rows; full training refits on 150 training categories before the 50-category test endpoint. Development caches follow their separate 100/50 class split. Do not use the main cache as a development-fold cache, and do not replace frozen text features with arbitrary embeddings while claiming numerical replication.

Additional-subject semantic rows need explicit WordNet/stimulus ID remapping and HDF5 verification. Those alignment results were recorded in the experiment, but the server queue that combined preprocessing and shutdown operations was deliberately not made a public launcher. Supply validated subject-specific caches explicitly.

Raw fMRI, original images, preprocessing artifacts and checkpoints are not included in Git. Archive permitted large artifacts separately and document hashes/provider links. No public dataset DOI exists in this candidate.
