# Reusable Case 3 motion inputs

Keep this directory when archiving or deleting old `results`. These copies
preserve the motion donors used by ordinary complete-model evaluations.
The study controllers have been removed.
These are fitted numerical inputs, not executable code.

| Folder | Donor |
| --- | --- |
| `single/` | Case 3, VR-parameterized across VR 3-8 |
| `two_tandem/` | Case 3, jointly VR-L/D-parameterized across VR 5-14 and L/D 1.5, 2, 3, 3.5, 4 |

Both donors use state Poly1/input Poly2, vanilla STLSQ (alpha=0, threshold=0.1,
normalization off), `overall_reduced` training, finite differences and the
dataset-specific Butterworth filter. Their coefficients were copied unchanged
from the original root results on 2026-11-01. `manifest.json` records the original
locations and SHA-256 hashes of both CSVs and metadata files.

Each `training.json` contains complete library/configuration snapshots and
finds `model.csv` beside itself. Old source paths inside historical metadata
are provenance; the originals are not needed to evaluate these copies.
The manifest uses relative paths so this input directory can move with a
checkout or an HPC transfer. The recorded hashes allow callers to verify the
identity of these inputs independently of deleted studies.

For a complete-model suite, pass `--case_3_model` with either donor
`training.json`, or an explicitly chosen compatible replacement. Do not overwrite these pinned copies or edit
their manifest to bypass a mismatch. Per-trajectory parameter substitution and
exact kinematic identities are still applied by the complete-model evaluator.

Raw CFD measurements are separate external inputs. Keep them available at the
locations configured in `case_bryan/constants.py` when using the ordinary
fitting/evaluation entry points.
