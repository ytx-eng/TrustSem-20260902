# Router CHANGELOG  (TrustSem revision report 2026-09-19)

This changelog tracks the specific code-level changes introduced in the
new-design Router bundle.  Read this alongside the bundle README and
`0_v17_core/router_migration_notes.md`.

## Phase 1: New module (`models/router/`)

### Added (5 files)

| File | Lines | Purpose |
|---|---|---|
| `__init__.py` | 30 | exports SubbandRouter, Shrinkage, LearnedThreshold, OnlineRouter, ProcessingConfig + legacy alias SubbandProcessor |
| `router.py` | 220 | SubbandRouter (R_psi) with 3-way softmax over `(m^pre, m^sup, m^id)`.  d_subband=16, d_packet=55, d_band_embed=16, hidden=64x2, T_m=1.0 learnable. |
| `shrinkage.py` | 175 | S_b(c) real (`sign(c)·max(0,|c|-λ)`) + complex (`c·max(0, 1-λ/(|c|+ε))`).  Parameter-free module. |
| `threshold.py` | 130 | LearnedThreshold with softplus parameterization, `init_lambda=0.05, eps_safe=1e-4`.  Always strictly positive. |
| `processing.py` | 285 | OnlineRouter wrapper.  R1 formula: `c_bar = (m^pre + m^id) c + m^sup S_b(c, lambda)`.  Also exports `compute_cbar` static method for testing. |

### Bug fix discovered during integration

* `router.py:_packet_descriptor` originally returned 32-dim
  (`mean + std` only) but `cfg.d_packet=55`.  This caused the router's
  own self-test to silently fail (mlp in_dim=87 vs actual 64).  The
  script-level self-tests had never been run with default cfg, so this
  was invisible until the integration pass.
* Fix: extended to `mean(16) + std(16) + max(16) + 7 global
  packet-level stats (L2/L1/energy/max_abs/mean_abs/std_abs/median_abs)`
  = 55-dim, matching `cfg.d_packet=55`.

## Phase 2: New training pipeline (`training/`)

### Added (3 files)

| File | Lines | Purpose | Maps to revision report |
|---|---|---|---|
| `stage_pretrain_shared.py` | 470 | frozen JSCC + warm-start router + threshold + head | §7.12 step 1+2 |
| `stage_router_train.py` | 480 | joint router + threshold + head fine-tune + reliability-weighted real/gen CE; freeze Θ_base* | §7.12 step 3-5 |
| `stage5_adapter_only.py` | 320 | freeze head C_omega + router R_psi + threshold; only fine-tune adapter A_nu with J_reg | §7.14 |

### Bug fix during integration

* `stage5_adapter_only.py` initially generated synthetic x with shape
  `(B, 1, 256)` (1D signal) but `RFEncoder` expects `(B, 1, 66, 66)`
  (2D image-like).  Changed shape to 66x66 for sim-mode testing.

## Phase 3: Deprecation markers (5 files, docstring-level only)

| File | What was deprecated | Why |
|---|---|---|
| `transitive_auth/__init__.py` | TransitiveAuthGraph, path_confidence, run_subband_analysis | §7.13 deletes receiver affinity + path-product + best-path weights |
| `losses/transitive_loss.py` | TransitiveAuthLoss (L_TA, Eq. 39), SeparationLoss (L_sep, Eq. 40) | §7.13 deletes the receiver-graph-based smoothing |
| `losses/generation_loss.py:TransitiveAuthGenLoss` | L_TA-gen, Eq. 51's transitive-auth term | §7.13 + §7.15: only keep real-anchor + reliability-weighted generated |
| `losses/generation_loss.py` (other classes) | DiffusionLoss, IdentityPreserveGenLoss, StyleMMDLoss | KEPT -- still used by stage3_generator_train |
| `training/stage2_graph_build.py` | receiver-graph construction (Eqs. 33-34) | §7.13 deletes receiver graph |
| `training/stage4_joint_train.py` | 10-loss joint training (paper §8 Eq. 57) | §7.12 mandates new optimization order; §7.13 deletes TA losses |
| `training/stage5_fewshot_adapt.py` | K-shot adapt that fine-tunes head + adapter simultaneously | §7.14 says only update nu_r_t, freeze head |

### What was NOT removed

* `transitive_auth/*.py` files are kept physically; only docstring
  banners added.
* `losses/transitive_loss.py`, `losses/generation_loss.py` files are
  kept; only TransitiveAuthGenLoss class is marked deprecated (other
  classes in `generation_loss.py` are still used by stage3).
* `stage4_joint_train.py` and `stage5_fewshot_adapt.py` are kept so the
  figure7 baseline + Figure 9 ablation comparisons still produce
  numbers.

Rationale: figure7 + Figure 9 are the only direct evidence we have for
the new design's improvement claims.  Removing the baseline scripts
would force us to re-derive those numbers from scratch.  Keeping them
behind deprecation banners gives a clear migration path without
breaking existing comparisons.

## Self-test results

All 7 self-tests pass on first try (after the two bug fixes above):

```
[OK] SubbandRouter  单测通过 (simplex + nonneg + shape)
[OK] Shrinkage      单测通过 (real + complex + broadcast + gradient)
[OK] LearnedThreshold 单测通过 (init / gradient / step / per-band / state_dict)
[OK] OnlineRouter   单测通过 (real + complex + simplex + 3 branches + grad + state_dict)
[OK] stage_pretrain_shared self-test passed
[OK] stage_router_train    self-test passed   (with §7.15 N_g=0 degeneration)
[OK] stage5_adapter_only   self-test passed   (with head_frozen=True)
```

## Design-level decisions (locked in)

* **k cannot be a router input feature** (per §7.2).  Documented
  loudly in every relevant docstring.  NO code-level assertion --
  per project agreement, design-level constraint is enough.
* **m^pre + m^sup + m^id == 1** (simplex-constrained softmax output).
* **lambda_sup,b > 0 strictly** (softplus parameterization, eps_safe=1e-4).
* **shrunken output is differentiable in (m, lambda) and broadcasts
  over the 16-subband axis without python loops**.
* **N_g=0 => generator loss defined as 0**, no division-by-zero
  (per §7.15).  Verified in `stage_router_train.py` self-test
  (`n_gen_used=0 -> ce_gen=0.0`).
* **OnlineRouter's R1 formula never crosses device boundaries** --
  each packet's decision depends only on its own subband statistics.
  No cross-receiver smoothing in the new design (per §7.6).

## Future work (not in this bundle)

* Real-data end-to-end run with stage1 + stage3 ckpts.
* Quantitative comparison figure7 (new design vs legacy baseline).
* Section IV v2 integration with new router (cache routing-aware
  features).
* Push to `ytx-eng/TrustSem-20260902` -- this bundle IS the v1 push.