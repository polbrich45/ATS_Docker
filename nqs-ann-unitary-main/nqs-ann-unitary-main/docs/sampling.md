# Model Sampling Theory, Frequency Born Draws, and Gumbel Top-K Beam Search

This document provides the theoretical formulation and mathematical proofs for the sampling algorithms used by the codebase. For full derivations, see [`theory.md`](theory.md).

---

## 1. Autoregressive Sampling Theory

### Wavefunction Factorization
The spin Autoregressive Neural Quantum State (NQS) factorizes the complex log-wavefunction site-by-site:

$$
\log \phi_\omega(s) = \frac{1}{2} \sum_{j=1}^N \log p_\omega(s_j \mid s_{<j}) + i \sum_{j=1}^N \varphi_\omega(s_j \mid s_{<j}),
$$

where $p_\omega(s_j \mid s_{<j})$ is the normalized conditional Born probability and $\varphi_\omega(s_j \mid s_{<j})$ is the conditional phase increment. The joint Born probability distribution $p_\omega(s) = |\phi_\omega(s)|^2$ is:

$$
p_\omega(s) = \prod_{j=1}^N p_\omega(s_j \mid s_{<j}).
$$

#### Proof of Exact Autoregressive Born Normalization
Summing $p_\omega(s)$ over all computational configurations $s \in \{0, 1\}^N$:

$$
\sum_{s_1, \dots, s_N} \prod_{j=1}^N p_\omega(s_j \mid s_{<j}) = \sum_{s_1} p_\omega(s_1) \sum_{s_2} p_\omega(s_2 \mid s_1) \dots \sum_{s_N} p_\omega(s_N \mid s_{<N}) = 1 \cdot 1 \dots 1 = 1.
$$

Because each conditional $\sum_{\sigma \in \{0,1\}} p_\omega(s_j = \sigma \mid s_{<j}) = 1$ is normalized, sequential autoregressive sampling draws **exact, uncorrelated Born samples** directly without Markov Chain Monte Carlo (MCMC) burn-in or autocorrelation time.

---

## 2. Frequency-Ranked Born Sampling (`SubspaceSampling.FREQUENCY`)

Frequency-ranked Born sampling draws a hard-capped batch of $N_{\text{seed\_samples}}$ (automatic device/basis-size budget, defined below) Born draws with replacement:

1. Draw $N_{\text{seed\_samples}}$ independent samples $s^{(a)} \sim p_\omega(s)$.
2. Compute empirical histogram counts $n(s) = \sum_{a=1}^{N_{\text{seed\_samples}}} \mathbf{1}_{s^{(a)} = s}$.
3. Rank unique configurations by empirical frequency descending.
4. Return up to $K$ top unique configurations.

---

## 3. Gumbel Top-K Beam Search (`SubspaceSampling.GUMBEL`)

Gumbel top-$K$ sampling draws **exactly $K$ unique configurations without replacement** directly from $p_\omega(s)$ using a stochastic beam search of width $K$ (Kim et al., [arXiv:2408.07625](https://arxiv.org/abs/2408.07625)).

At each site $j$, child Gumbel scores are conditioned on the parent Gumbel score $g_P$:

$$
g_{\text{child}} = -\log \left( e^{-g_P} + e^{-\text{raw}} - e^{-\max(\text{raw})} \right).
$$

The implementation in `src/nqs.py` uses a fixed-width JAX scan. One call returns $K$ unique sector-valid configurations, or raises if numerical failure violates that contract. It evaluates at most $K$ prefixes per site and never retries or enumerates the full support. The Transformer recomputes full-sequence conditionals for each prefix batch; this is not a cached incremental decoder, and JAX compilation does not guarantee a single fused GPU kernel. Increasing $K$ changes compiled shapes and increases attention and beam memory.

The underlying algorithm is [ancestral Gumbel top-$K$ sampling](https://jmlr.org/papers/v21/19-985.html), applied to autoregressive NQS in [Malyshev, Schmitt, and Lvovsky](https://arxiv.org/abs/2408.07625). Its ordered draws obey the sequential without-replacement law. For configurations $a$ and $b$ with Born probabilities $p(a)$ and $p(b)$, respectively,

$$
P(x_1=a,x_2=b)=p(a)\frac{p(b)}{1-p(a)},\qquad a\ne b\;.
$$

Unique configurations construct Ritz subspaces. Ordinary VMC and reported sampled energies still use independent Born draws with replacement; averaging local energies uniformly over a unique sample would generally be biased. Tests check the nonuniform ordered-pair law, not only uniqueness and the $K=1$ marginal.

## 4. SQD-like solve budgets and coverage

The selected-space solve uses two independent budgets. `RotationConfig.n_subspace` caps the number $M$ of unique configurations in the Ritz matrix; `n_seed_samples` caps the number $B$ of with-replacement Born draws used to discover them. Increasing $B$ at fixed $M$ improves discovery statistics but does not remove selected-space truncation. The Hamiltonian is assembled exactly on the retained space, so a Ritz eigenvalue does not acquire a Monte Carlo standard error from the discovery count. Accuracy still depends on the chosen basis.

Omitted values resolve when the configuration is constructed. The automatic matrix cap is 128 on CPU or 256 on GPU/TPU. For frequency sampling, the draw heuristic is

$$
B=\max(32M,B_{\min})\;,\qquad B_{\min}=2048\ \text{on CPU},\quad B_{\min}=8192\ \text{on GPU/TPU}\;.
$$

Explicit positive integer budgets override the policy, including smaller draw caps. Device selection respects `jax.default_device`. The notebook keeps the 3 x 3 basis below the full 512-state Hilbert space and doubles the automatic matrix cap for its 4 x 4 profile.

| Notebook profile | Device | Unique Ritz-state cap $M$ | Born-draw cap $B$ |
| --- | --- | ---: | ---: |
| 3 x 3 | CPU | 128 | 4096 |
| 3 x 3 | GPU/TPU | 256 | 8192 |
| 4 x 4 | CPU | 256 | 8192 |
| 4 x 4 | GPU/TPU | 512 | 16384 |

Each outer summary records `sampling`: the method, requested caps, actual Born draws, unique observed configurations before ranking truncation, ranked seed candidates, retained states, unique Gumbel draws, final basis size, and `nqs_mass_before`. The last quantity is the exact NQS probability sum on the constructed basis before rotation and teacher updates, not a frequency estimate or ground-state overlap. Gumbel mode reports zero Born draws. It obtains $M$ unique seeds directly; its unused Born-draw setting is not charged as actual sampling.

A peaked empirical batch can contain fewer than $M$ unique states. Retention, Hamiltonian-connected completion, and at most one bounded Gumbel call then fill the matrix; they do not trigger more Born draws. Gumbel draws are reported separately. Retained probability mass helps diagnose inadequate coverage, but even large mass does not guarantee a small energy error or better subsequent NQS training. Check larger matrix caps and the physical post-teacher/recovery energy when assessing convergence. The paired 64-state audit in `validation.md` remains a result for its recorded budget; increasing these defaults does not establish a rotation advantage.
