# Sampling-informed orbital optimization

For $N$ spatial orbitals, the Hubbard NQS has a fixed number of particles in each spin block. A real spatial matrix $R(\theta)$, parameterized by angles $\theta$, acts equally on both spins. OpenFermion rotates the one- and two-body integrals with $R$, and NetKet supplies the fermionic VMC operator. Every conversion uses `convert_spin_blocks=True` because OpenFermion interleaves spins and NetKet stores consecutive spin blocks. The physical state and its Hamiltonian in the NQS representation obey

$$
|\Psi\rangle=\mathcal U(R)|\phi_\omega\rangle,\qquad H_R=\mathcal U(R)^\dagger H\mathcal U(R)\;.
$$

Here $\omega$ denotes NQS parameters and $\mathcal U(R)$ is the particle-conserving second quantization of $R$. A circuit layer has an even-bond sublayer followed by an odd-bond sublayer along the spatial-orbital order. Each bond has one independent real Givens angle. For $L$ layers, there are $L(N-1)$ angles and the matrix is the ordered product $R=G_0G_1\cdots$. This replaces the previous all-pair exponential parameterization. Older fermionic checkpoints with that parameterization must not be reused; start a new run. No Fock-space unitary is stored.

## Objectives and adaptation

At each outer iteration, unique autoregressive samples seed a selected determinant set $S$ of size $M$. Retained determinants and Hamiltonian-connected completion fill the chosen cap. The projected Hamiltonian $H_S(R)$ is diagonalized to obtain its normalized lowest eigenvector $c$ and energy $E_S$. The IPR is $I=\sum_{x\in S}|c_x|^4$, and normalized participation entropy is $s_p=-\log I/\log M$. For the zero-extended selected state, let $\rho_k$ denote its reduced density operator on the first $k$ spatial orbitals, with both spins included. The normalized cut-entropy proxy is

$$
s_e=\frac{-\sum_{k=1}^{N-1}\log\operatorname{Tr}\rho_k^2}{\sum_{k=1}^{N-1}\log\max\{2,\min(M,4^{\min(k,N-k)})\}}\;.
$$

`RotationConfig.entanglement_order="spatial"` uses these grouped cuts and their fermionic permutation signs. The alternative `"model"` uses the NQS's consecutive spin-block mode order. In that case, $\rho_k$ refers to the first $k$ modes, the sums run from $1$ to $2N-1$, and the normalization replaces $4^{\min(k,N-k)}$ with $2^{\min(k,2N-k)}$. No mode permutation is applied. This includes the up/down cut, whose exact full-state Schmidt spectrum is invariant under separate rotations of the two spin blocks; truncation can change it. Both choices are representation proxies rather than guarantees of autoregressive learnability. The Hubbard CLI and benchmark expose `--entanglement-order`; spin runs retain their existing site-cut criterion.

The configured objective is

$$
\mathcal L(R)=\frac{E_S(R)}{2N\max(|t|,|U|)}+\lambda_p s_p(R)+\lambda_e s_e(R)\;,
$$

where $t$ and $U$ are the Hubbard hopping and interaction, while signed coefficients $\lambda_p$ and $\lambda_e$ are `participation_weight` and `entanglement_weight`. Positive coefficients favor compactness; zero coefficients disable the corresponding entropy calculation. `orbital_ritz_objective` exposes this calculation independently of the loop. Eigenvectors can be discontinuous at degeneracy; these objectives are heuristics on a truncated state, and finite-difference L-BFGS-B is a local search. `n_rot` caps optimizer iterations rather than the number of objective evaluations, which is reported separately.

The final Ritz probabilities and relative phases teach the existing NQS; VMC then recovers at fixed new angles. This is approximate Ritz-guided adaptation, not exact rotation of the old NQS. Energy acceptance uses the adapted NQS and restores its parameters and orbital angles on rejection. A fresh final estimate avoids reporting the noisy estimate used to select an accepted proposal. The recorded `subspace_mass` measures the NQS Born weight on $S$; selected-state entropies do not control omitted amplitudes.

The depth-zero control ansatz has a specific restriction. Write $u$ and $d$ for the up- and down-spin occupation strings in the NQS order, and $u_N$ for the final up-spin occupation. Without attention blocks, each conditional head sees only the preceding occupation and its position; the particle-count prior resets at the spin-block boundary. Its amplitude therefore factors as $\phi_\omega(u,d)=f_\omega(u)g_{\omega,u_N}(d)$, with $f_\omega$ and $g_{\omega,u_N}$ complex functions. The Schmidt rank across the up/down cut is at most two. Separate spatial rotations on the two spin blocks preserve this rank, so these rotations cannot remove that ansatz restriction. They can still change other cuts and the optimization landscape. This bound does not apply to the attention-enabled ansatz and is not an energy-floor calculation.

## Cost and related approaches

The implementation uses NetKet for VMC/direct autoregressive sampling, Optax for NQS and teacher updates, OpenFermion for integral rotations, and SciPy for orbital search. A cached list of selected Slater-Condon transitions contracts trial integrals into an $M\times M$ matrix. It avoids rebuilding a VMC operator at every finite-difference trial. Sparse Schmidt matrices use at most $O(M^2)$ workspace per cut. Integral storage is still $O(N^4)$, selected diagonalization is $O(M^3)$, and preparation of the transition list is polynomial in $M$ and $N$; the plan is cached per determinant basis, so it is not recomputed when the accepted basis is unchanged. This is intended for compact research experiments, not large active spaces. Exact enumeration is restricted to the existing small-sector oracle.

Circuit depth also controls operator cost. Each disjoint nearest-neighbor sublayer has bandwidth one, so $L$ even/odd layers obey $R_{ij}=0$ for $|i-j|>2L$, where $i,j\in\{1,\ldots,N\}$ label the orbital-chain order. An onsite Hubbard interaction therefore has at most $\min\{N^4,N(4L+1)^4\}$ distinct spatial-index combinations after combining equal terms. This is a support bound, not a runtime estimate; the current integral arrays remain dense. Deeper circuits can improve the selected-state objective while making every subsequent NQS update more expensive. The six-site comparison in [the validation record](validation.md) measures this tradeoff directly.

[Li's entanglement-minimized orbitals](https://arxiv.org/abs/2506.13386) use spin-adapted MPS, local Renyi-1/2 minimization, and randomized global orbital search. The present spatial Renyi-2 proxy is inspired by representation simplification but does not reproduce that algorithm. Energy-based orbital optimization in [SQD](https://qiskit.github.io/qiskit-addon-sqd/guides/optimize_orbitals.html) provides a complementary criterion. These objectives can disagree: a compact selected state may be an inferior teacher. Therefore the benchmark judges post-adaptation NQS energy, with all outer work included in its cost, rather than treating compactness or Ritz improvement as a learning result.


## Local and global orbital search

`RotationConfig.n_basin=0` uses one SciPy L-BFGS-B solve. A positive `n_basin` adds that many [SciPy basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) moves, each followed by the same local solve. Global displacements have angular scale $\pi/2$, and the Metropolis temperature is 0.1 in the dimensionless objective units above. Both are heuristic search settings. A seeded generator derived from the loop's checkpointed JAX key makes repeated runs and restarts reproducible. All objective evaluations and all search time count toward the comparison budget. This uses the basin-hopping idea discussed by [Li](https://arxiv.org/abs/2506.13386), without reproducing the paper's MPS local sweeps or discrete swap layers. The spin loop rejects `n_basin` because its rotation optimizer is different.

The `--n-basin` option is available in the Hubbard CLI and the paired benchmark. SciPy >=1.15 is required for its generator-based randomness API. A more globally optimized representation surrogate is not automatically a better NQS target: the five-seed warm-stationarity probe rejected all ten global-search proposals, despite improved Ritz-state compactness. See the validation record for the retained controls.
