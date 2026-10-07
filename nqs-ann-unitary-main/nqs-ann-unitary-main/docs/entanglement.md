# Projected-State Rényi-2 Entanglement Entropy

This document specifies the theoretical formulation, reduced density matrix derivation, compilation mechanics, and scientific limitations of the sampled Rényi-2 entanglement objective ($\bar{S}_2$). For full derivations, see [`theory.md`](theory.md) and [`loss.md`](loss.md).

---

## 1. Theoretical Formulation & Reduced Density Matrix

For a retained sampled computational basis $S_M = \{s_i\}_{i=1}^M$ and lowest Ritz eigenvector $v_0 \in \mathbb{C}^M$, define the zero-extended state $|\psi_S\rangle$ on the full Hilbert space:

$$
\psi_S(s) = \begin{cases} v_{0,i}, & s = s_i \in S_M, \\ 0, & s \notin S_M. \end{cases}
$$

For each ordered spin cut $A_k = \{1, \dots, k\}$ ($k = 1, \dots, N-1$), partition the spin indices into subsystem $A_k$ and environment $B_k = \{k+1, \dots, N\}$. Any basis state decomposes as $s_i = (a_i, b_i)$ where $a_i \in \{0, 1\}^k$ and $b_i \in \{0, 1\}^{N-k}$.

The reduced density matrix $\rho_{A_k} = \operatorname{Tr}_{B_k} |\psi_S\rangle \langle \psi_S| / \|\psi_S\|^2$ has matrix elements:

$$
(\rho_{A_k})_{a, a'} = \sum_{b} v_{0, (a, b)} v_{0, (a', b)}^*,
$$

where the sum runs over all environment configurations $b \in \{0, 1\}^{N-k}$ present in $S_M$. The purity of subsystem $A_k$ is:

$$
\operatorname{Tr} \rho_{A_k}^2 = \sum_{a, a'} |(\rho_{A_k})_{a, a'}|^2.
$$

The normalized sum of ordered-cut Rényi-2 entanglement entropies is:

$$
\bar{S}_2 = \frac{\sum_{k=1}^{N-1} -\log \operatorname{Tr} \rho_{A_k}^2}{\sum_{k=1}^{N-1} \log \min(M, 2^{\min(k, N-k)})}.
$$

---

## 2. Compilation Mechanics & $O(N M^2)$ Scaling

Computing $\bar{S}_2$ reuses a single $M^2$ scatter buffer across all $N-1$ cuts via `jax.lax.scan` inside [`compute_renyi2_entropy`](file:///home/klimak/Codes/2026/nqs-ann-unitary/src/rotation_ritz.py#L453) in [`src/rotation_ritz.py`](file:///home/klimak/Codes/2026/nqs-ann-unitary/src/rotation_ritz.py).

- Pair index matrices are precomputed once during [`prepare_rotation_problem`](file:///home/klimak/Codes/2026/nqs-ann-unitary/src/rotation_ritz.py#L340).
- Time complexity is $O(N M^2)$, avoiding dense $2^N$ matrix operations.

---

## 3. Scientific Limitations & MPS Proxy Interpretation

> [!CAUTION]
> **Subspace Truncation Artifact**: $\bar{S}_2$ measures the entanglement of the *zero-extended sampled state* $|\psi_S\rangle$, not the full physical quantum state $|\Psi\rangle$. If weight outside $S_M$ is non-negligible, $\bar{S}_2$ provides an estimate of sampled basis entanglement rather than a global MPS bond dimension bound.
