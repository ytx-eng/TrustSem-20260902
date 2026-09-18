# TrustSem Router Bundle v1  (2026-09-19)

This bundle contains the **new-design TrustSem Router module** that replaces
the legacy `transitive_auth/` graph architecture, per the chapter revision
report (downloaded separately as `TrustSem_chapter_revision_report.md`).

## What's in this bundle

```
TrustSem-RouterBundle-v1/
├── README.md                                # this file
├── ROUTER_CHANGELOG.md                      # what changed in this revision
├── TrustSem_chapter_revision_report.md      # source document for the redesign
├── TrustSem_SectionIV_Full_Experiment_Prompt.md   # Section IV prompt
└── 0_v17_core/                              # entire v17 core (code + docs only,
                                             #  data / checkpoints / experiments
                                             #  excluded -- see "Data" below)
    ├── router_migration_notes.md            # new vs old design mapping
    ├── models/router/                       # NEW: SubbandRouter + Shrinkage +
    │                                        #       LearnedThreshold + OnlineRouter
    │   ├── __init__.py
    │   ├── router.py
    │   ├── shrinkage.py
    │   ├── threshold.py
    │   └── processing.py
    ├── training/
    │   ├── stage_pretrain_shared.py         # NEW: §7.12 step 1+2
    │   ├── stage_router_train.py            # NEW: §7.12 step 3-5
    │   ├── stage5_adapter_only.py           # NEW: §7.14
    │   ├── stage4_joint_train.py            # DEPRECATED (figure7 baseline)
    │   ├── stage5_fewshot_adapt.py          # LEGACY BASELINE
    │   └── stage2_graph_build.py            # DEPRECATED
    ├── losses/
    │   ├── transitive_loss.py               # DEPRECATED
    │   └── generation_loss.py               # TransitiveAuthGenLoss deprecated,
    │                                        # others (DiffusionLoss, etc.) kept
    ├── transitive_auth/__init__.py          # DEPRECATED
    └── ... (rest of v17 core, unchanged)
```

Total bundle size: **< 1 MB** (code + docs only).  Data files (.h5, .npz,
.pth, .csv) and experiment outputs are excluded -- see the next section.

## Data

The original v17 dataset (`Wisig ManyTx`) and intermediate artifacts are
**NOT** included in this bundle due to size.  To run the new pipeline you
need:

| Data | Location | Notes |
|---|---|---|
| Raw WSF / Wisig ManyTx samples | `D:/无人机/2024.8/manytx/data` | Per the v17 stack; raw I/Q per device, multi-day, multi-rx. |
| Stage 1 / 2 / 3 / 4 ckpts | `D:/tmp/.../checkpoints/` | Produced by the existing stage1-4 scripts. |
| Optional: Section IV cache | `D:/tmp/trustsem_cache/` | 900 records × 4 subdirs = 3600 .npy files. |

The new pipeline scripts run in `--mode sim` without any of these
(self-test + synthetic descriptors).  For real-data runs you need
stage1/stage3 ckpts to be present.

## What's NEW in this revision (post §7.13-§7.16)

1. **`models/router/`** -- the entire subband router module:
   - `SubbandRouter` (R_psi): per-subband 3-way softmax weights
   - `Shrinkage` (S_b): real + complex soft-thresholding operator
   - `LearnedThreshold` (lambda_sup,b): per-band learnable non-negative threshold
   - `OnlineRouter`: R1 formula wrapper `c_bar = (m^pre + m^id) c + m^sup S_b(c, lambda)`
2. **`training/stage_pretrain_shared.py`** -- §7.12 step 1+2:
   load frozen JSCC backbone, warm-start router + threshold + head triplet
3. **`training/stage_router_train.py`** -- §7.12 step 3-5:
   joint router+threshold+head fine-tune with reliability-weighted
   real/generated CE; freeze shared params to emit final `Theta_base*`
4. **`training/stage5_adapter_only.py`** -- §7.14:
   freeze head + router + threshold; only fine-tune the adapter `A_nu`
   with `J_reg` (L2 distance from initial params)

## What's DEPRECATED (kept ONLY for figure7 baseline + Figure 9 ablation)

- `transitive_auth/` (receiver graph)
- `losses/transitive_loss.py` (L_TA + L_sep)
- `losses/generation_loss.py:TransitiveAuthGenLoss` (L_TA-gen)
- `training/stage2_graph_build.py`
- `training/stage4_joint_train.py` (figure7 baseline only)
- `training/stage5_fewshot_adapt.py` (legacy baseline)

These files have docstring-level deprecation banners but still import
cleanly so the figure7 + Figure 9 baselines keep producing numbers.

## Self-test commands

From inside `0_v17_core/`:

```bash
# Module-level self-tests (no real data needed)
python -m models.router.router
python -m models.router.shrinkage
python -m models.router.threshold
python -m models.router.processing

# Training pipeline self-tests
python -m training.stage_pretrain_shared --self_test
python -m training.stage_router_train    --self_test
python -m training.stage5_adapter_only   --self_test
```

Each one prints a final `[OK]` line if all invariants hold.

## End-to-end pipeline (sim mode, no real data)

```bash
cd 0_v17_core

python -m training.stage_pretrain_shared --mode sim      # → stage_pretrain_shared.pth
python -m training.stage_router_train    --mode sim      # → stage_router_train.pth (frozen)
python -m training.stage5_adapter_only   --mode sim      # → stage5_adapter_only.pth + kshot JSON
```

For real data, replace `--mode sim` with `--mode real` and ensure the
stage1 / stage3 ckpts + Wisig ManyTx data are at the expected paths.