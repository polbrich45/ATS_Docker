# Selected-Space Completion and Epstein-Nesbet PT2

This document details the boundary of the sampled Ritz subspace, the Epstein-Nesbet importance ranking, and the second-order perturbative correction ($\Delta E_2$). For full derivations, see [`theory.md`](theory.md).

---

## 1. Selected and External Spaces

Let $S_M = \{s_i\}_{i=1}^M$ be the retained computational basis. The projected Hamiltonian satisfies $H_{S_M}(\theta) v_0 = \epsilon_0 v_0$.

The external frontier $\mathcal{F}$ contains connected configurations $y \notin S_M$ with non-zero off-diagonal coupling to $v_0$:

$$
A_y = \langle y | H_\theta | v_0 \rangle = \sum_{i \in S_M} \langle y | H_\theta | s_i \rangle v_{0,i}.
$$

Pauli paths are combined before testing whether $A_y$ vanishes. In particular, cancelled $XX$ and $YY$ spin-flip paths must not generate frontier states. For a fixed-magnetization NQS, selection and fallback filling explicitly enforce the same spin sector.

---

## 2. Epstein-Nesbet Importance Selection

When completing a subspace up to target dimension $M$, external configurations $y \in \mathcal{F}$ are ranked by their coherent Epstein-Nesbet importance:

$$
\eta_y = \frac{|\langle y | H_\theta | v_0 \rangle|^2}{H_{yy} - \epsilon_0}.
$$

States with $H_{yy} - \epsilon_0 \le 10^{-9}$ (near-crossings) receive top priority ($\eta_y \to \infty$).

---

## 3. Second-Order Epstein-Nesbet Correction ($\Delta E_2$)

The second-order Epstein-Nesbet correction $\Delta E_2$ is:

$$
\Delta E_2 = \sum_{y \notin S_M}^{H_{yy} - \epsilon_0 > 10^{-9}} \frac{|\langle y | H_\theta | v_0 \rangle|^2}{\epsilon_0 - H_{yy}}.
$$

#### Proof of Nonvariational Nature
While the Ritz ground energy obeys $\epsilon_0(S_M, \theta) \ge E_{\text{exact}}$ by the Rayleigh-Ritz variational principle, the PT2 correction $\Delta E_2 \le 0$ can overshoot the exact ground state energy near subspace level crossings ($H_{yy} - \epsilon_0 \approx 0$).

> [!CAUTION]
> The PT2-corrected energy $\epsilon_0 + \Delta E_2$ is **not variational** and has no strict ordering bound. It serves as a diagnostic for missing frontier weight and must never be presented as a variational energy bound.

---

## 4. Frontier Bounds and Truncation

Frontier aggregation is capped at `max(4096, 64 * M)` states. If the frontier size exceeds this bound, the implementation sets `frontier_truncated = 1.0` and reports `pt2 = NaN` to avoid presenting an incomplete sum as the true PT2 correction.
