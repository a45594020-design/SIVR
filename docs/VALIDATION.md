# Validation and scope

Executed locally on 2026-10-26:
- Seven CPU tests passed (paths, aggregates, duplicate pooling/gradient, source-control invariants, three retained development tests).
- 85-unit/255-row aggregate means and sample SDs recomputed successfully.
- Research source files syntax-compiled; public command help and environment inventory checked.

Runtime versions are in local_test_versions.json. Full dependency installation in a clean environment, CUDA training, end-to-end raw-data preparation, full numerical checkpoint replay, and additional-subject preprocessing were not executed in this packaging task. Local h5py/tensorboardX/timm are absent; requirements-training.txt declares them. The package includes an archived backbone plus upstream fallback support files, not a verified Git commit of the running server. This limitation is documented in third_party/README.md and backbone_provenance.json.

Training code changes: relative internal imports; configurable paths; the existing response-control adjacency appended by the original queue now lives directly in priors.py. Development examples are illustrative configurations. No metrics were edited or new experimental results generated.
