# SIVR

Structure-Informed Brain–Visual–Linguistic Learning for Visual Category Decoding.

This repository organizes the existing research implementation and recorded experiment aggregates for the TMM manuscript. The proposed addition is a decoder-side graph regularizer on the inherited BraVL backbone. Fine-grained structure means relations between unique measured voxels. The main endpoint is visual category decoding; decoder outputs are feature vectors, not generated images.

## What is available and tested

- `scripts/reproduce_tables.py` recomputes means and sample SDs from 85 training units / 255 V,T,VT evaluation rows, using bundled aggregate data. This CPU command was executed successfully.
- Seven CPU tests check path resolution, aggregate completeness, inverse PCA with duplicate pooling, graph-loss gradients, batch replication invariance, source-control graph properties, and development-module contracts.
- The archived training modules are packaged with relative imports and externally configurable paths. The S-perm/S-rewire implementation originally appended by an extension queue is included directly in `src/sivr/replication/priors.py`.
- The backbone snapshot and upstream support files are included with provenance and the upstream MIT license.

**Full GPU training has not been rerun from this organized package.** The local validation runtime used CPU PyTorch. Raw fMRI, frozen feature inputs, preprocessing caches, pretrained visual weights and checkpoints are not bundled. See `docs/DATA.md` and `docs/VALIDATION.md`. The files are a research release candidate, not a claim of newly reproduced performance.

## Repository layout

```text
src/sivr/
  paths.py                 external path configuration
  replication/             corrected main protocol, priors, single-run trainer, preparation
  development/             E4/E5 residual, modality and auxiliary-evaluation code
configs/
  paths.example.json       copy to paths.local.json and edit paths
  replication/             frozen protocol and nine seed-0 condition configurations
  development/             protocol records and illustrative seed-0 configurations
scripts/
  reproduce_tables.py      CPU recomputation from included aggregates
  check_environment.py     package and input-path inventory
  run_experiment.py        one configured experiment, no queue operations
  prepare_data.py          archived sub04 data/feature preparation entry
  generate_replication_configs.py  five-seed configuration expansion
  evaluate_saved.py        development checkpoint evaluation wrapper
  run_cached_contracts.py  optional real-cache replication checks (requires CUDA)
tests/                     seven CPU regression/contract checks
third_party/bravl/         archived backbone + upstream support files and MIT license
data/aggregates/           machine-readable existing results
requirements.txt           lightweight aggregate-analysis environment
requirements-training.txt  additional research/training dependencies
requirements-cpu-tested.txt  exact directly used versions in local CPU checks
environment.yml            candidate conda environment, not original server lockfile
docs/                      data schema, provenance, validation and release checklist
```

## 1. Recompute tables without GPU or raw fMRI

Use Python 3.10 or later, from this repository root:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_tables.py
```

Output: `results/summary_verified.csv`. The script validates the recorded unit count, class of reference spaces and aggregate arithmetic. It does not verify individual predictions or recompute bootstrap intervals from raw predictions.

## 2. Install the training dependencies and run CPU contracts

Install the appropriate PyTorch/torchvision build for your platform and GPU first, then:

```bash
python -m pip install -r requirements-training.txt
python -m pip install -e .
python -m pytest -q
python scripts/check_environment.py
```

`environment.yml` is an alternative starting point. It is not the experiment server's frozen environment. The older upstream environment is recorded separately in `docs/upstream_environment_historical.yml`; do not combine its Python 3.7 pins blindly with this Python 3.10 package.

## 3. Configure external paths

Copy `configs/paths.example.json` to `configs/paths.local.json` and edit the paths. Relative paths are resolved relative to the JSON file, not the shell working directory. `paths.local.json` is gitignored.

```bash
python scripts/check_environment.py --paths configs/paths.local.json
```

Set a separate output root for each subject. Cached brain/semantic rows, dimensions and voxel identities determine the scientific input domain. Additional-subject preprocessing and ID remapping must be validated before interpreting another-subject run; this package does not silently prepare all five subjects.

## 4. Run a configured main-protocol experiment

After preparing or supplying the required fitted cache:

```bash
python scripts/run_experiment.py --family replication --paths configs/paths.local.json --config configs/replication/B0_seed0.json --out runs/B0_seed0 --phase final
python scripts/run_experiment.py --family replication --paths configs/paths.local.json --config configs/replication/S_seed0.json --out runs/S_seed0 --phase final
```

The launcher checks cache files and CUDA availability and installs the recorded frozen configuration in the configured output root. Do not reuse an output directory with a different config. The inherited runner reseeds after legacy imports and records initial-weight hashes.

Nine conditions are supplied: B0, R, S, G, S+G, G-perm, G-rewire, S-perm, S-rewire. Expand seed-0 examples for the five paired seeds:

```bash
python scripts/generate_replication_configs.py --out runs/configs
```

Graph seed is `10000 + model seed`. Frozen coefficients are preserved. The five-seed configuration expansion is not a command to start 45 GPU jobs; select and run jobs explicitly.

## 5. Preprocessing and development experiments

`python scripts/prepare_data.py --paths configs/paths.local.json` wraps the archived sub04 fitting/feature pipeline. It requires the original HDF5/MAT input layout, image files and frozen RepVGG weights. It fits class-specific preprocessing and uses CUDA eigendecomposition. `SIVR_GOD_H5` and `SIVR_TEST_IMAGES` can override the HDF5/test-image locations. See `docs/DATA.md` before running.

```bash
python scripts/run_experiment.py --family development --paths configs/paths.local.json --config configs/development/B0_seed0.json --out runs/dev_B0_seed0 --phase fold1
```

Development configurations are **illustrative entry examples**, not recovered per-run JSONs. Compare every parameter with the archived protocol and actual frozen cache before attempting numerical replication. E4/E5 uses a separate first development fold and auxiliary Ridge/cosine retrieval. It must not be pooled with VT-SVC main results. The optional saved-checkpoint wrapper is documented by `python scripts/evaluate_saved.py --help`.

## Interpretation of included results

Main sub04 VT means: B0 Top-1 7.03%, G 6.97%, R 7.89%. No source-specific Top-1 advantage is established. Five subjects were trained separately, not in a cross-subject transfer setting. Archived X0–X4 results have a legacy initialization limitation; corrected replication and E4/E5 records are labeled separately. Stored bootstrap intervals are included as recorded outputs.

## Data, citation and licensing

Original GOD source: https://kamitanilab.github.io/GenericObjectDecoding/ . Source-provider terms apply. Do not redistribute raw third-party images/fMRI/pretrained models without the relevant permission.

Credit the BraVL paper and preserve `third_party/bravl/LICENSE` when reusing its code. See `third_party/README.md` for snapshot provenance. The authors still need to choose a license for their own SIVR additions and derived data; no blanket license has been invented for those materials. See `docs/RELEASE_CHECKLIST.md` before public release.

No GitHub URL, Zenodo DOI or final SIVR publication citation is asserted until actually created. This repository does not include server credentials or cloud shutdown/cleanup queues.
