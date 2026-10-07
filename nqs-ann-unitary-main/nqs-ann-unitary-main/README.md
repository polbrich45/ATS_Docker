# Adaptive-Basis Neural Quantum States

This repository implements a block-coordinate variational method that combines Autoregressive Neural Quantum States (NQS) with shallow unitary basis rotations and a direct Ritz-to-NQS state transfer teacher.

---

## Workflow Overview

The method alternates between NQS VMC training, shallow circuit optimization on a retained Ritz subspace, and direct state transfer:

$$
H_\theta = U_\theta^\dagger H U_\theta, \qquad |\Psi_{\omega,\theta}\rangle = U_\theta |\phi_\omega\rangle.
$$

```text
for AdaptiveStep in AdaptiveConfig.steps:
    train warm-started NQS on H_theta via VMC/minSR
    refine shallow circuit U_theta on retained Ritz subspace
    solve final sampled Ritz problem H_{S_M}(theta)
    transfer joint Ritz probabilities and relative phases to NQS
    recover NQS by short VMC training at fixed new angles
    accept using NQS energy; otherwise restore pre-rotation NQS and angles
```

Rotation optimizes the sampled Ritz objective, while acceptance compares the actual NQS before rotation and after recovery. Exact enumeration is used for small reference systems. With sampled energies, the spin loop permits an increase within `acceptance_sigma` combined standard errors; the fermionic loop requires a decrease exceeding that many errors (default `2`). The spin tolerance is a safeguard, not evidence of an energy gain. `final_energy` and each outer row's `energy` describe the accepted state; `energy_after_rotation`, `energy_after_teacher`, and `energy_after_recovery` describe the proposal even when rejected.

---

## Documentation Roadmap

- **[`docs/theory.md`](docs/theory.md)**: Main definition-complete mathematical reference.
- **[`docs/method.md`](docs/method.md)**: Compact workflow overview.
- **[`docs/validation.md`](docs/validation.md)**: Paired spin experiments, fermionic checks, limitations, and next research tests.
- **[`docs/loss.md`](docs/loss.md)**: Exact objective definitions and scale normalizations.
- **[`docs/basis.md`](docs/basis.md)**: Circuit geometry, Pauli conjugation, and sampling rules.
- **[`docs/sampling.md`](docs/sampling.md)**: Model sampling, frequency-ranked Born sampling, and Gumbel top-$K$ beam search.
- **[`docs/teacher.md`](docs/teacher.md)**: Ritz-to-NQS teacher loss derivation and HI-NQS adaptation.
- **[`docs/entanglement.md`](docs/entanglement.md)**: Projected-state Rényi-2 entanglement entropy construction.
- **[`docs/pt2.md`](docs/pt2.md)**: Subspace completion, Epstein-Nesbet PT2 correction, and nonvariational bounds.
- **[`docs/dmrg.md`](docs/dmrg.md)**: Optional final DMRG reference solver interface.

---

## Quick Start & CLI Example

[`notebooks/optimize.ipynb`](notebooks/optimize.ipynb) selects `SYSTEM="tfi"` (transverse-field Ising, `hx=0.7`, `hz=0.03`) or `SYSTEM="j1j2"` ($J_2/J_1=0.5$, fixed $S^z$). Neither model is given a positive or Marshall sign prior; the network learns its phase, with $S=\sigma/2$. `QUICK=True` uses open 3 x 3 or 4 x 2 lattices with exact diagnostics; `QUICK=False` uses periodic 4 x 4. The outer loop is staged: warm VMC, rotation stages with a doubling Ritz basis, and a fixed-basis polish. Rotation adds model-specific Rényi-2 and nonstoquasticity penalties to the Ritz energy, and an optional section runs the spinful Hubbard orbital loop. Acceptance uses the physical NQS energy after teaching and recovery; a lower Ritz value alone does not establish improvement. The CLI example below retains the J1-J2 model.

`TrainConfig()` selects 256 independent samples on CPU and 1024 on GPU/TPU. Explicit `n_samples` or CLI `--n-samples` values override this choice, including when CPU is explicitly selected on a GPU machine. Defaults use constant learning rate 0.02, SR diagonal shift 0.01, and [minSR with SPRING momentum 0.8](https://netket.readthedocs.io/en/stable/api/_generated/driver/netket.driver.VMC_SR.html). SR directions are not clipped by default; first-order training retains norm-1 clipping. CPU minSR uses explicit Jacobians, while accelerators use the lazy NTK path. These are modern optimization defaults, not a claim of benchmark superiority. Several defaults changed relative to earlier revisions (learning rate 0.002 -> 0.02, SR shift 1e-4 -> 1e-2, SR momentum 0 -> 0.8, SR clip 1 -> 0, and device-dependent `n_samples`/`n_subspace`/`n_seed_samples`); pass explicit values to reproduce runs pinned to the previous defaults.

The Transformer uses Flax key/value caches for exact autoregressive sampling while retaining the full causal network for energy evaluation and differentiation. Caches are fresh for every draw batch and are not saved as model parameters. A single sparse reference solve is available through 16 spins, while automatic full-basis diagnostics stop at 12 spins. Larger NQS evaluations use Monte Carlo energy checks with uncertainty. If a full-space exact reference already exists, a requested final exact reference reuses its original-basis energy rather than diagonalizing the expanded rotated operator again.

The SQD-like Ritz solve has separate [discovery and matrix budgets](docs/sampling.md#4-sqd-like-solve-budgets-and-coverage). The 3 x 3 notebook uses 4096 Born draws and 128 unique Ritz states on CPU, or 8192 draws and 256 states on GPU/TPU. Explicit `n_seed_samples` and `n_subspace` values override these heuristics. Outer summaries report actual draws, unique observations, bounded Gumbel completion, and retained NQS mass before teaching.

The notebook runs the actual adaptive loop for several outer steps and uses `deps/general_plot.Plotter` directly. Separate figures show physical NQS energy through training, angle updates, teaching and continued VMC; energy variance; rotation and teacher loss histories; and sampled-space coverage. Whole-loop controls compare VMC alone, teaching without rotation, energy-only rotation, and IPR/Renyi-2 penalties from the same initial seed and configured VMC budgets. Energy comparisons report accepted states; the target measures also retain rejected proposals. Figures are saved as PNG and PDF. Timings and steady VMC slowdown are recorded, and the controls are not claimed to have equal total cost. In the three-seed 3 x 3, 64-state [matched-cost audit](docs/validation.md#ising-rotation-teacher-and-runtime-audit-2026-10-01), rotations increased VMC cost and did not outperform continued VMC or the unrotated teacher control.

Logs use the vendored `deps/general_log` library with literal tab indentation, stage banners, physical NQS energies and uncertainties, SQD coverage, teacher losses, proposal decisions, and per-outer timing tables. Training tail energies are labeled as pre-update Monte Carlo observations. Timings include setup and compilation; steady VMC slowdown excludes warmup. Initialize the dependencies with `git submodule update --init --recursive`, then install the logger as a package with `pip install -e deps/general_log`; `pip install -r requirements.txt` includes this step. Set `NQS_PROGRESS=0` to hide progress bars, `PYLOGCOLORS=0` for plain text, and optionally `PYLOGFILE=/path/to/run.log` to also save messages. `display="silent"` suppresses project messages and progress bars.

```bash
python run.py \
  --lx 4 --ly 4 --pbc --j1 1.0 --j2 0.5 --hx 0.0 \
  --conserve-total-sz --n-up 8 \
  --n-outer 4 --n-nqs 1000 --n-rot 300 \
  --n-samples 1024 --n-subspace 512 \
  --f-retain 0.1 --nested-weight 0.0 \
  --subspace-sampling frequency --n-seed-samples 4096 \
  --family givens --n-layers 4 \
  --n-teacher 50 --n-recovery 20 --output-dir runs/j1j2_ritz
```

---

## Python API Example

```python
from src.adaptive import run_adaptive_loop
from src.config import (
    AdaptiveConfig,
    AdaptiveStep,
    CircuitConfig,
    FinalConfig,
    LatticeConfig,
    ModelConfig,
    RotationConfig,
    TeacherConfig,
    TrainConfig,
)

lattice = LatticeConfig(Lx=4, Ly=4, pbc=True, j2=0.5)
model = ModelConfig(width=64, depth=3, heads=4, conserve_total_sz=True, n_up=8)

step = AdaptiveStep(
    train=TrainConfig(n_nqs=800, n_samples=1024),
    rotation=RotationConfig(
        n_rot=200,
        n_subspace=512,
        f_retain=0.1,
        nested_weight=0.0,
        participation_weight=0.0,
        entanglement_weight=0.0,
        nonstoquasticity_weight=0.0,
    ),
    teacher=TeacherConfig(n_teacher=50),
    n_recovery=20,
)

adaptive = AdaptiveConfig(
    circuit=CircuitConfig(n_layers=4, family=("givens",)),
    steps=(step, step),
    final=FinalConfig(solver="none"),
    display="plain",
)

result = run_adaptive_loop(lattice, model, adaptive, seed=0, output_dir="runs/adaptive_run")
```

---

## Testing & Validation

```bash
pytest -q
```

For the paired six-spin ablation (continued VMC, teacher only, rotation only, full loop), run `python -m scripts.benchmark_basis --ablation --n-samples 128 --output-dir runs/comparison`. Each seed uses one common warm checkpoint. Results and per-phase timings are saved to `ablation.json` and the individual run folders. This matches VMC update budgets, not total computational cost; see [`docs/method.md`](docs/method.md) for the interpretation and next research checks.

Install the optional `fermion` extra, then run `python -m scripts.run_hubbard` for the spinful Hubbard NQS loop. Configure the lattice with `--Lx`, `--Ly`, and `--pbc`, and set `--n-up`, `--n-down`, `--t`, and `--U` independently of the model and training budgets. An even-site lattice defaults to half filling in each spin block; odd-site lattices require explicit counts. The complex autoregressive transformer uses NetKet's direct sampler with exact particle-count masks. Each step trains, optimizes spatial-orbital angles on a sampled Ritz subspace, fits the teacher, recovers with VMC, and accepts only on an energy check. A sector of at most 64 determinants uses exact energy as an oracle; larger sectors use sampled energy with an uncertainty threshold and never enumerate all determinants. `--restart-from` resumes a saved outer checkpoint. This is a tunable fermionic workflow, not a benchmarked state-of-the-art ansatz or the entanglement-minimized-orbital algorithm of [Li](https://arxiv.org/abs/2506.13386). The exact orbital-rotation conventions are checked by `tests/test_basis_research.py::test_short_orbital_circuit_and_cached_projection`.


## Equal-cost research benchmark

`python -m scripts.benchmark_basis --output-dir runs/budget-spin --seeds 0 1 2` compares the full outer loop with ordinary VMC from the identical warm NQS. `vmc_updates` matches attempted VMC updates; `vmc_time` spends the measured adaptive wall time entirely on VMC, including its driver setup and compilation. The adaptive budget includes sampling, Ritz, rotation, teaching, recovery, evaluation, and checkpoint I/O. Ordinary VMC retains its initial state and the best checkpoint evaluated every `--checkpoint-every` updates, then reports a fresh final estimate. Monitoring counts toward its time budget. There is no subspace solve in either plain baseline. `--teacher-control` also runs teaching and recovery with zero basis rotations, reporting its energy and cost to check whether rotations add value. Run sequentially on an idle machine. Timings include compilation, and the baseline runs after the adaptive branch, so cache reuse can favor ordinary VMC. Budgets stop after completed updates; the output reports the actual cost ratio and whether a safety cap prevented reaching the time budget.

Use `--system hubbard --Lx 2 --Ly 2 --n-subspace 12` for the fermionic comparison. `--width`, `--depth`, and `--phase-head-depth` set the same NQS in both branches. The benchmark saves fresh final energies, paired differences and uncertainties, per-phase timings, attempted updates, and accepted proposals. A lower Ritz energy or an update-matched win does not establish an equal-cost learning gain. Plateau escape remains a research hypothesis until repeated runs from demonstrably plateaued states support it.

The Hubbard loop now uses `CircuitConfig.n_layers` short even/odd nearest-neighbor spatial Givens layers, with the same orbital matrix for both spins. `participation_weight` penalizes normalized $-\log(\mathrm{IPR})$ of the selected Ritz state; `entanglement_weight` penalizes normalized cut Renyi-2 entropy. `entanglement_order="spatial"` groups both spins of each orbital with fermionic permutation signs; `"model"` follows the NQS's spin-block mode order. These are compactness proxies within the selected support. See [`docs/orbitals.md`](docs/orbitals.md) for conventions, costs, and the distinction from entanglement-minimized MPS orbital optimization.


The measured compact Hubbard example wins against the best retained equal-time ordinary-VMC checkpoint in all three tested seeds (mean total-energy difference -0.79047); the width-8 spin example loses all three equal-time comparisons. These are small-sector feasibility results with deliberately limited NQS capacity and compilation-inclusive CPU timings. Systematic plateau escape and scaling gains remain unproven. Reproduction commands and controls are recorded in [`docs/validation.md`](docs/validation.md).


The benchmark also supports `--polish-warmup N` for a small-sector stationarity diagnostic using exact NQS energy gradients and SciPy trust-region optimization. It is restricted to at most 64 configurations and 256 real parameters. Gradient and Hessian checks are reported explicitly and do not certify global minima or stability along flat directions. This shared preparation precedes every comparison branch. `--teacher-control` now spends its remaining adaptive time budget on ordinary unrotated VMC, retaining the teacher endpoint when continuation worsens it; retained baseline weights are saved alongside histories. `--n-basin N` enables optional fermionic basin hopping inspired by the orbital-search work in [`docs/orbitals.md`](docs/orbitals.md).

The stronger five-seed warm-stationarity probe finds one numerically stationary escape example: energy -1.000000 becomes -1.573153 while both equal-time controls retain -1.000000. Only two of five seeds gain overall, and four warm states fail the strict stationarity checks. Optional global orbital search improves the compactness surrogate but gives no accepted gain in that cohort. These results keep systematic plateau escape an open research question.

Repeating that probe with model-order entropy also improves two of five seeds, with the stationary seed reaching -1.401407. Neither cut ordering is consistently better. The depth-zero control has up/down Schmidt rank at most two, a restriction that separate spin-block orbital rotations preserve. Its gains demonstrate conditional optimization benefits rather than full ground-state representability; details and raw-result locations are in [`docs/validation.md`](docs/validation.md).

Global energy-based orbital search (`--n-basin 3`) with an attention-enabled NQS and 24 of 36 determinants improves over equal-time plain VMC and an unrotated SQD teacher in all five tested seeds, including two additional seeds evaluated with the configuration frozen. Higher-rate plain-VMC controls also lose in every seed. Four adapted runs finish within 0.01 of the exact ground energy -2.102748. This demonstrates repeatable rotation-specific gains for the tested $2\times2$ Hubbard protocol. Compactness penalties and deeper circuits can hurt: in a six-site sampled-energy cohort, the basis loop beats ordinary VMC but its unrotated teacher has lower point estimates throughout. The paired seed-0 depth comparison reduces outer-loop time from 98 to 36 seconds with similar energy within uncertainty. See [`docs/validation.md`](docs/validation.md) for all controls, timing, limitations, and reproducible commands.
