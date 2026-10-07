# Method: Train, Rotate, Teach

For full mathematical derivations, NQS architecture, variational estimators, circuit conjugation, eigensystem derivatives, and teacher equations, see [`theory.md`](theory.md). This document presents the workflow design, theoretical comparisons, and literature context.

---

## 1. Theoretical Motivation: Why Decouple Basis and Sampler?

Circuit angles $\theta$ and NQS weights $\omega$ can be optimized jointly or alternately. The present method uses a deterministic projected eigensolve for the angle block. It removes Monte Carlo gradient noise conditional on the chosen subspace, but does not eliminate subspace-selection error or guarantee escape from local minima.

We decouple the optimization into a **block-coordinate adaptive loop**:

$$
H_\theta = U_\theta^\dagger H U_\theta, \qquad |\Psi_{\omega,\theta}\rangle = U_\theta |\phi_\omega\rangle.
$$

1. **NQS VMC / minSR Step**: The NQS $|\phi_\omega\rangle$ is optimized on the rotated Hamiltonian $H_\theta$ via sample-space stochastic reconfiguration (minSR), finding the best variational state in the current reference frame.
2. **Sampled Ritz Refinement**: A finite computational basis $S_M$ ($M = 256 \dots 1024$) is constructed using retained states ($f_{\text{retain}} = 0.1$), fresh Born/Gumbel samples, and Epstein-Nesbet connected states. Circuit angles $\theta$ are refined via **exact differentiable Rayleigh-Ritz eigensolves** on $H_{S_M}(\theta)$, avoiding Monte Carlo gradient variance entirely during basis updates.
3. **Ritz-to-NQS Teacher**: Transfer joint Ritz probabilities and relative phases. Diagonal-energy and uniform log-probability biases are optional and default to zero; their combined coefficients must satisfy the boundedness check in [`teacher.md`](teacher.md).
4. **Recovery and acceptance**: Run `n_recovery` VMC updates at fixed proposed angles. If spin recovery raises energy beyond the combined-error tolerance, restore its initial NQS state; retain both endpoint measurements and all recovery cost. Compare the selected proposal with an independent evaluation immediately before rotation. The spin loop rejects worsening beyond its configured uncertainty tolerance; the fermionic loop requires a decrease exceeding its combined-error threshold. Rejection restores NQS weights and angles while keeping the random stream advanced. Checkpoints and retained Ritz weights refer to the accepted basis.

---

## 2. Literature Map & Structural Comparisons

[HI-NQS](https://arxiv.org/abs/2606.26760) motivates iterative sampling, selected diagonalization, and distillation. The present spin implementation teaches the joint Ritz distribution and adds a relative-phase loss.

[Entanglement-minimized orbitals](https://arxiv.org/abs/2506.13386) uses spin-adapted MPS, orbital Givens rotations, and randomized exploration. Its energy check follows wavefunction reoptimization. This supports testing alternating basis proposals and recovery, but does not establish an NQS advantage. The present projected-state Renyi-2 penalty differs from that paper's Renyi-1/2 MPS objective.

---

## 3. Subspace Retention & Refresh Rules ($f_{\text{retain}}$)

To prevent subspace instability across outer adaptive steps:
1. **Subspace Freezing ($f_{\text{retain}} = 0.1$)**: Retains $K_{\text{retain}} = \lfloor f_{\text{retain}} \cdot M \rfloor$ states from the previous Ritz subspace $S_{\text{prev}}$, ranked by previous Ritz vector probabilities $|v_{0,s}|^2$.
2. **Fresh Seed Sampling**: Draws fresh ANN seeds via **Frequency-Ranked Born Sampling** (default) or **Gumbel Top-K Beam Search**.
3. **Epstein-Nesbet Connected Completion**: Ranks coupled, sector-valid states by $\eta_y = |\langle y | H_\theta | v_0 \rangle|^2/(H_{yy} - \epsilon_0)$. Zero connections after Pauli cancellation are excluded. Nonpositive denominators are treated as crossings rather than perturbative estimates.
4. **Gumbel Fallback**: Fills any remaining budget shortfall up to $M$.

## 4. Minimal validation experiment

Use a small frustrated lattice with an exact reference first. Start each comparison from identical warmed-up NQS weights and zero angles. Compare continued VMC, teacher only, rotation only, and the full loop over several paired seeds. Keep VMC updates, samples, architecture, and recovery budgets equal; record the additional rotation and teacher time. Repeat comparisons at matched wall time before claiming efficiency.

Report accepted NQS energy and variance, proposal energy changes, acceptance rate, fixed-subspace Ritz gain, operator term count, and elapsed time. For Monte Carlo-sized systems, use fresh final samples independent of acceptance to avoid selection bias. A lower Ritz energy with unchanged or worse NQS energy is not a successful transfer. Better energy after extra recovery alone does not establish a basis advantage.

The first research variation should be a small frozen nested-subspace weight, not several simultaneous penalties. Participation entropy, projected entanglement, and sampled sign structure measure different properties; none is a proven proxy for Transformer trainability. If proposals fail mainly at teaching, test longer recovery or a normalized wavefunction-overlap teacher before making the circuit deeper. Randomized symmetry-preserving angle restarts are a later basin-escape experiment, with the same recovery and acceptance check.

NetKet already supplies the direct sampler, variational state, VMC/minSR driver, and optimizers. Its native autoregressive networks are useful unconstrained baselines. Retain the current conditional fixed-magnetization mask: the native [ARDirectSampler](https://netket.readthedocs.io/en/stable/api/_generated/samplers/netket.sampler.ARDirectSampler.html) does not support a constrained Hilbert space directly. Replacing the network alone would not preserve this sampling contract.

For fermions, `tests/test_basis_research.py::test_short_orbital_circuit_and_cached_projection` is the exact convention check, and [`src/fermionic.py`](../src/fermionic.py) implements the loop. A production extension should rotate one- and two-body tensors and use NetKet's fermionic operators rather than expand general orbital rotations through the spin Pauli pipeline. Occupation ordering and separate up/down particle-number masks must be explicit. Reuse the alternating loop and teacher mathematics only after these support and sign contracts are tested.
