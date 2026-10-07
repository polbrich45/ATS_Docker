# Adaptive-Basis Neural Quantum States

## Variational formulation, sampled Ritz rotation, and direct state transfer

### Abstract

We formulate a block-coordinate variational method in which an autoregressive neural quantum state (NQS) and a fixed-depth unitary circuit are optimized as distinct parameter blocks. The NQS represents and samples the ground state of the rotated Hamiltonian $H_\theta = U_\theta^\dagger H U_\theta$. A sampled Rayleigh-Ritz problem then refines a retained basis under a hard cap and optimizes the normalized Ritz energy alongside auxiliary basis metrics. A direct teacher transfers joint Ritz probabilities and relative phases to the NQS. Fixed-angle VMC recovery precedes acceptance by the NQS energy; diagonal and uniform log-probability biases are optional and disabled by default. The circuit is refined rather than extended, so the outer iteration does not increase quantum depth. The method is variational at the NQS and Ritz levels, whereas a reported Epstein-Nesbet correction is explicitly nonvariational. This document defines the wavefunction convention, circuit action, objectives, derivatives, sampling rules, diagnostics, and failure criteria used by the implementation.

---

## I. Notation and Basis Convention

Let $N$ spin-$1/2$ degrees of freedom span a computational basis $\{|s\rangle\}$, where $s = (s_1, \dots, s_N) \in \{0, 1\}^N$ denotes a bit string ($s_j = 0$ for spin-up $\uparrow$, $s_j = 1$ for spin-down $\downarrow$). The physical Hamiltonian $H$ is represented as a Pauli string expansion.

$$
H = \sum_{P \in \mathcal{P}_N} c_P P, \qquad c_P \in \mathbb{C}, \qquad H = H^\dagger\;,
$$

where $\mathcal{P}_N = \{I, X, Y, Z\}^{\otimes N}$ is the $N$-qubit Pauli group and $c_P$ are complex expansion coefficients.

The variational ansatz decouples into two parameter blocks.

$$
|\Psi_{\omega,\theta}\rangle = U_\theta |\phi_\omega\rangle, \qquad H_\theta = U_\theta^\dagger H U_\theta\;,
$$

where $\omega$ denotes all NQS parameters, $\theta$ denotes the variational angles of a fixed shallow circuit $U_\theta$, $|\phi_\omega\rangle$ is the state represented by the NQS in the rotated reference frame, $H_\theta$ is the rotated Hamiltonian, and $|\Psi_{\omega,\theta}\rangle$ is the physical quantum state in the lab frame.

The corresponding Rayleigh quotients yield equivalent energy functionals.

$$
E(\omega, \theta) = \frac{\langle \Psi_{\omega,\theta} | H | \Psi_{\omega,\theta} \rangle}{\langle \Psi_{\omega,\theta} | \Psi_{\omega,\theta} \rangle} = \frac{\langle \phi_\omega | H_\theta | \phi_\omega \rangle}{\langle \phi_\omega | \phi_\omega \rangle}\;.
$$

By unitary invariance, the exact spectrum of $H_\theta$ is identical to $H$, i.e., $\operatorname{spec}(H_\theta) = \operatorname{spec}(H)$. Basis optimization cannot alter the exact ground-state eigenvalue; its purpose is to transform the computational basis so that a restricted NQS or a finite sampled Ritz subspace captures the ground state with lower truncation error, reduced participation entropy, and simplified sign structure.

One outer adaptive iteration performs the block-coordinate sequence.

$$
\omega \xrightarrow[H_\theta]{\text{VMC/minSR}} \omega' \xrightarrow[\omega' \text{ fixed}]{\text{sampled Ritz}} \theta' \xrightarrow[H_{\theta'}]{\text{Ritz teacher}} \omega''\;.
$$

The displayed sequence produces a proposal. Fixed-angle VMC recovery follows teaching. The spin loop retains the state before recovery if the measured increase exceeds its combined-error tolerance; both measured endpoints and all recovery cost remain reported. The selected proposal's physical NQS energy is then compared with its pre-rotation value. The spin loop rejects a worsening proposal beyond its Monte Carlo uncertainty tolerance by restoring both parameter blocks. The fermionic loop instead requires a decrease exceeding the configured combined-error threshold, as described in [the orbital notes](orbitals.md). Both accepted parameter sets $(\omega, \theta)$ are warm-started across iterations while the circuit topology and angle count remain fixed. The Ritz objective is a surrogate; its improvement alone does not establish improvement of the NQS.

---

## II. Autoregressive Neural Quantum State

### A. Amplitude and Phase Parameterization

The NQS $|\phi_\omega\rangle$ is parameterized by an Autoregressive Transformer ([`ARTransformer`](../src/model.py#L105) in [`src/model.py`](../src/model.py)) with pre-layer-normalized residual blocks and dual output heads. For a bit string $s = (s_1, \dots, s_N)$, the complex log-wavefunction factorizes site-by-site.

$$
\log \phi_\omega(s) = \frac{1}{2} \sum_{j=1}^N \log p_\omega(s_j \mid s_{<j}) + i \sum_{j=1}^N \varphi_\omega(s_j \mid s_{<j})\;,
$$

where $s_{<j} = (s_1, \dots, s_{j-1})$, $p_\omega(s_j \mid s_{<j})$ is the normalized conditional Born probability, and $\varphi_\omega(s_j \mid s_{<j})$ is the conditional phase increment. The amplitude head outputs logits $a_{j, \sigma}$ for $\sigma \in \{0, 1\}$, giving

$$
p_\omega(s_j = \sigma \mid s_{<j}) = \frac{e^{a_{j,\sigma}}}{e^{a_{j,0}} + e^{a_{j,1}}}\;.
$$

Consequently, the joint Born probability $p_\omega(s) = |\phi_\omega(s)|^2 = \prod_{j=1}^N p_\omega(s_j \mid s_{<j})$ is strictly normalized over the computational basis.

$$
\sum_{s \in \{0,1\}^N} |\phi_\omega(s)|^2 = \sum_{s \in \{0,1\}^N} \prod_{j=1}^N p_\omega(s_j \mid s_{<j}) = 1\;.
$$

The phase head alters the complex phase without affecting the conditional Born probabilities or normalization.

### B. Magnetization Sector Masking ($U(1)$ Symmetry)

When total magnetization $S^z_{\text{tot}} = \frac{1}{2} \sum_{j=1}^N Z_j$ is conserved (`ModelConfig.conserve_total_sz=True`) with $n_\uparrow$ up spins, invalid conditional choices are masked at each site $j$. Let $r = N - j$ be the remaining unassigned sites and $m$ be the remaining required up spins. The untrained prior probability is hypergeometric.

$$
p(s_j = 0 \mid s_{<j}) = \frac{m}{r+1}, \qquad p(s_j = 1 \mid s_{<j}) = \frac{r+1-m}{r+1}\;.
$$

The learned logits $a_{j,\sigma}$ are added to $\log p(s_j = \sigma \mid s_{<j})$ prior to the softmax. At zero initialization, the prior produces a uniform distribution over all $\binom{N}{n_\uparrow}$ configurations in the sector. Fixed-sector sampling is admitted only when $[H, S^z_{\text{tot}}] = 0$ and $[G_g, S^z_{\text{tot}}] = 0$ for all circuit generators $G_g$.

### C. Born Sampling and Variational Monte Carlo

Because each site conditional $p_\omega(s_j \mid s_{<j})$ is normalized, independent samples $s^{(a)} \sim p_\omega(s)$ are drawn sequentially from site $1$ to $N$ without Markov Chain Monte Carlo (MCMC) burn-in or autocorrelation time.

The variational Monte Carlo (VMC) estimator of the rotated energy $E_\omega = \langle \phi_\omega | H_\theta | \phi_\omega \rangle$ from $N_s$ independent Born samples $\{s^{(a)}\}_{a=1}^{N_s}$ is

$$
\widehat{E} = \frac{1}{N_s} \sum_{a=1}^{N_s} E_{\text{loc}}(s^{(a)}), \qquad E_{\text{loc}}(s) = \sum_{s'} \langle s | H_\theta | s' \rangle \frac{\phi_\omega(s')}{\phi_\omega(s)}\;.
$$

The sample variance of $E_{\text{loc}}(s)$ measures the distance from an exact eigenstate (zero-variance principle).

### D. Stochastic Reconfiguration and Quantum Geometry

For NQS parameters $\omega$, logarithmic derivatives are $O_k(s) = \partial_{\omega_k} \log \phi_\omega(s)$. The Fubini-Study metric tensor on the quantum state manifold (the Quantum Geometric Tensor, QGT) is

$$
S_{kl} = \langle O_k^* O_l \rangle - \langle O_k^* \rangle \langle O_l \rangle\;.
$$

Stochastic Reconfiguration (SR) solves the regularized natural gradient equation.

$$
\sum_l (\operatorname{Re} S_{kl} + \lambda \delta_{kl}) \delta \omega_l = -\eta F_k, \qquad F_k = 2 \operatorname{Re}[\langle O_k^* E_{\text{loc}} \rangle - \langle O_k^* \rangle \langle E_{\text{loc}} \rangle]\;.
$$

#### Minimum-Step SR (minSR) Formulation

Let $X \in \mathbb{R}^{2N_s \times P}$ be the centered real parameter Jacobian matrix evaluated over $N_s$ Monte Carlo samples (accounting for real and imaginary parts). The sample-space Neural Tangent Kernel (NTK) is $K = X X^T \in \mathbb{R}^{2N_s \times 2N_s}$. The minSR update is

$$
\delta \omega = -\eta X^T (X X^T + \lambda I)^{-1} e_{\text{loc}}\;,
$$

where $e_{\text{loc}} \in \mathbb{R}^{2N_s}$ contains the centered local energies. Solving $K v = e_{\text{loc}}$ via Cholesky decomposition costs $O(N_s^3)$ time and $O(N_s^2)$ memory, which scales independently of the parameter count $P$.

---

## III. Shallow Unitary Circuit and Generator Normalization

### A. Circuit Structure and Spectral Normalization

The unitary circuit $U_\theta$ is a product of exponentiated Hermitian generators.

$$
U_\theta = \prod_{g=1}^{N_g} e^{-i \theta_{\pi(g)} G_g}, \qquad G_g = G_g^\dagger\;,
$$

where $N_g$ is the number of gates and $\pi(g)$ maps gate $g$ to its parameter index in $\theta$. Every generator is normalized to spectral norm $\|G_g\|_2 = 1/2$, matching the scale of single-spin operators $S^a_j = \frac{1}{2} \sigma^a_j$.

### B. 2-Site and 3-Site Matchings (Givens and String Givens)

The standard 2-site exchange generators are

$$
G_{ij}^{\text{givens}} = \frac{X_i Y_j - Y_i X_j}{4}, \qquad G_{ij}^{\text{xy}} = \frac{X_i X_j + Y_i Y_j}{4}, \qquad G_{ij}^{\text{heis}} = \frac{X_i X_j + Y_i Y_j + Z_i Z_j}{6}\;.
$$

On square $J_1$-$J_2$ lattices, the 3-site `string_givens` generator acts on diagonal pairs $(i, j)$ through an intermediate nearest-neighbor site $k$.

$$
G_{ikj}^{\text{string}} = \frac{X_i Z_k Y_j - Y_i Z_k X_j}{4}\;.
$$

$G_{ikj}^{\text{string}}$ is Hermitian, real-orthogonal upon exponentiation, conserves total $S^z_{\text{tot}}$, and has spectral norm $1/2$. Disjoint 3-site matchings are constructed via [`string_givens_matchings`](../src/lattices.py#L266) in [`src/lattices.py`](../src/lattices.py).

### C. Analytic Pauli-String Conjugation and Operator Growth

For a single Pauli string $P \in \mathcal{P}_N$ and generator $G$ with $G^2 = I$.

$$
e^{i \alpha G} P e^{-i \alpha G} = \begin{cases} P, & [G, P] = 0, \\ P \cos(2\alpha) + i G P \sin(2\alpha), & \{G, P\} = 0. \end{cases}\;.
$$

Conjugating $H = \sum_P c_P P$ string-by-string yields $H_\theta = U_\theta^\dagger H U_\theta$. Terms with coefficients below `--theta-cutoff` (default $10^{-12}$) are pruned. Operator growth is monitored via the operator-growth metric.

$$
g_{\text{op}} = \frac{\sum_{P \notin \operatorname{supp}(H)} |c_P(\theta)|^2}{\sum_{P \in \operatorname{supp}(H)} |c_P(0)|^2}\;.
$$

---

## IV. Sampled Ritz Subspace Construction

### A. Subspace Refresh and Retained-State Selection ($f_{\text{retain}}$)

Let $M = \text{RotationConfig.n\_subspace}$ be the target Ritz subspace dimension. At each outer step, subspace freezing retains the top fraction $f_{\text{retain}}$ (default $0.1$, configured via `--f-retain`) of states from the previous Ritz subspace $S_{\text{prev}}$.

$$
K_{\text{retain}} = \min\left(|S_{\text{prev}}|, \lfloor f_{\text{retain}} \cdot |S_{\text{prev}}| \rfloor\right)\;.
$$

Retained states are ranked by their previous Ritz vector probabilities $|v_{0,s}|^2$ (with current NQS probabilities $|\phi_\omega(s)|^2$ as fallback).

### B. Frequency-Ranked and Gumbel Top-$K$ Seed Sampling

The remaining subspace slots are populated by fresh ANN configurations.

1. **Frequency-Ranked Sampling (`SubspaceSampling.FREQUENCY`, default)**: Draws $N_{\text{seed\_samples}}$ (automatic device/basis-size budget; see [`sampling.md`](sampling.md#4-sqd-like-solve-budgets-and-coverage)) Born samples with replacement, computes empirical frequencies, and ranks unique configurations.
2. **Gumbel Top-$K$ Sampling (`SubspaceSampling.GUMBEL`)**: Performs exact without-replacement top-$K$ beam search using top-down conditioned Gumbel noise. [`gumbel_top_k_configs`](../src/nqs.py#L205) guarantees returning **exactly $K$ unique, sector-valid configurations**.

### C. Hamiltonian-Connected Frontier Construction & Epstein-Nesbet Importance

If fresh ANN seeds and retained states do not fill $M$, the subspace is completed using Hamiltonian-connected states. For a provisional ground state $v_0$ on seed subspace $S_0$, external configurations $y \notin S_0$ coupled by $H_\theta$ form the frontier $\mathcal{F}$.

The coherent off-diagonal coupling amplitude is $A_y = \langle y | H_\theta | v_0 \rangle$, and states are ranked by coherent Epstein-Nesbet importance.

$$
\eta_y = \frac{|\langle y | H_\theta | v_0 \rangle|^2}{H_{yy} - \epsilon_0}\;.
$$

Near-crossings ($H_{yy} - \epsilon_0 \le 10^{-9}$) are admitted first with infinite priority. The top-$\eta_y$ configurations fill the remaining budget up to $M$.

### D. Projected Hamiltonian and Rayleigh-Ritz Variational Principle

For the selected $M$-dimensional basis $S_M = \{s_i\}_{i=1}^M \subset \{0,1\}^N$, the projected Hamiltonian matrix $H_{S_M}(\theta) \in \mathbb{C}^{M \times M}$ is

$$
(H_{S_M}(\theta))_{ij} = \langle s_i | U_\theta^\dagger H U_\theta | s_j \rangle\;.
$$

The lowest Ritz eigenpair $(E_{\text{Ritz}}, v_0)$ satisfies

$$
H_{S_M}(\theta) v_0 = \epsilon_0 v_0, \qquad v_0^\dagger v_0 = 1\;.
$$

By the Rayleigh-Ritz variational principle

$$
E_{\text{exact}} \le \epsilon_0(S_M, \theta)\;.
$$

### E. Differentiable Ritz Eigenvalue and Lorentzian-Regularized Vector Derivatives

The derivative of the ground Ritz energy $\epsilon_0$ with respect to rotation angles $\theta$ follows the Hellmann-Feynman theorem

$$
\frac{\partial \epsilon_0}{\partial \theta_k} = \operatorname{Re}\left( v_0^\dagger \frac{\partial H_{S_M}(\theta)}{\partial \theta_k} v_0 \right)\;.
$$

For state-dependent auxiliary objectives, the eigenvector derivative $\frac{\partial v_0}{\partial \theta_k}$ uses a Lorentzian-regularized JVP implemented in [`_ground_state_jvp`](../src/rotation_ritz.py#L403).

$$
\frac{\partial v_0}{\partial \theta_k} = \sum_{n > 0} v_n \frac{g_n}{g_n^2 + \delta^2} \left( v_n^\dagger \frac{\partial H_{S_M}(\theta)}{\partial \theta_k} v_0 \right)\;,
$$

where $g_n = \epsilon_0 - \epsilon_n$ and $\delta = 10^{-3} \max(\epsilon_{M-1} - \epsilon_0, J_{\text{rms}})$. This regularization remains well-behaved near subspace level crossings.

### F. Second-Order Epstein-Nesbet Perturbation Theory (PT2)

After angle optimization, the nonvariational Epstein-Nesbet PT2 energy correction is evaluated over external connected configurations $y \notin S_M$.

$$
\Delta E_2 = \sum_{y \notin S_M} \frac{|\langle y | H_\theta | v_0 \rangle|^2}{\epsilon_0 - \langle y | H_\theta | y \rangle}\;.
$$

The PT2-corrected energy $\epsilon_0 + \Delta E_2$ is **not variational** and serves as a diagnostic for subspace truncation quality.

---

## V. Rotation Objectives and Losses

### A. Dimensionless Scale Normalization ($N J_{\text{rms}}$)

To ensure invariance under positive energy rescaling $H \to c H$ ($c > 0$), we define the root-mean-square Hamiltonian scale.

$$
J_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{P \ne I} |c_P|^2}\;.
$$

The normalized Ritz energy is $\bar{E} = \frac{\epsilon_0}{N J_{\text{rms}}}$.

### B. Complete Rotation Loss Formulation

The complete differentiable rotation loss is

$$
\mathcal{L}_\theta = \frac{\epsilon_0}{N J_{\text{rms}}} + \lambda_{\text{part}} \bar{S}_{\text{part}} + \lambda_{\text{ent}} \bar{S}_2 + \lambda_{\text{sign}} g_{\text{sign}} + \lambda_{\text{nested}} L_{\text{nested}}\;,
$$

where:

- $\bar{S}_{\text{part}}$ is the normalized sampled participation entropy.
- $\bar{S}_2$ is the normalized all-cut Rényi-2 entanglement entropy of the zero-extended sampled state.
- $g_{\text{sign}}$ is the sampled nonstoquasticity indicator.
- $L_{\text{nested}}$ is the nested Ritz loss.

### C. Nested Ritz Objective ($L_{\text{nested}}$)

The nested Ritz loss evaluates the variational truncation penalty between the full subspace $S_M$ and its top half $S_{M/2} = \text{Top}_{M/2} \{|v_{0,s}|^2\}$.

$$
L_{\text{nested}} = \frac{\epsilon_{M/2} - \epsilon_M}{N J_{\text{rms}}}\;.
$$

Subspace indices $S_{M/2}$ are frozen during each rotation optimization block and refreshed between blocks. The weight satisfies $0\leq\lambda_{\text{nested}}\leq1$, so the combined energy terms form a convex interpolation of the full and smaller Ritz energies in the same per-site units.

### D. Subspace Participation Entropy ($\bar{S}_{\text{part}}$)

With $p_i = |v_{0,i}|^2 / \|v_0\|^2$ for $i \in \{1, \dots, M\}$.

$$
\bar{S}_{\text{part}} = \frac{-\log \sum_{i=1}^M p_i^2}{\log M}\;.
$$

$\bar{S}_{\text{part}} \in [0, 1]$ measures state compactness **within $S_M$ only**.

### E. Projected-State Rényi-2 Entanglement ($\bar{S}_2$)

For each ordered spin cut $A_k = \{1, \dots, k\}$, the zero-extended state reduced density matrix is $(\rho_{A_k})_{aa'} = \sum_b v_{0,ab} v_{0,a'b}^*$. The normalized cut entanglement sum is

$$
\bar{S}_2 = \frac{\sum_{k=1}^{N-1} -\log \operatorname{Tr} \rho_{A_k}^2}{\sum_{k=1}^{N-1} \log \min(M, 2^{\min(k, N-k)})}\;.
$$

### F. Sampled Nonstoquasticity Indicator ($g_{\text{sign}}$)

Sampled nonstoquasticity measures positive off-diagonal and imaginary matrix element weight in $H_{S_M}(\theta)$.

$$
g_{\text{sign}} = \frac{\sum_{i \ne j} \left[ \max(\operatorname{Re}(H_{S_M})_{ij}, 0)^2 + \operatorname{Im}(H_{S_M})_{ij}^2 \right]}{\sum_{i \ne j} |(H_{S_M})_{ij}|^2}\;.
$$

> [!CAUTION]
> $g_{\text{sign}} = 0$ indicates that the matrix elements within the sampled subspace $S_M$ are stoquastic. It is **not** a global stoquasticity certificate for the full Hilbert space.

---

## VI. Phase-Aware Ritz Teacher (State Transfer)

### A. Joint Spin Distribution & Spin Adaptation of HI-NQS

The Ritz ground vector $v_0$ defines target probabilities $q_i = |v_{0,i}|^2 / \|v_0\|^2$ over configurations $s_i \in S_M$. The teacher loss adapts HI-NQS for spin systems.

$$
\mathcal{L}_T = \frac{1}{N} D_{\text{KL}}(q \parallel p_\omega) + \frac{w_E}{N} \sum_{i=1}^M q_i A_i \log p_\omega(s_i) + \frac{w_H}{M N} \sum_{i=1}^M \log p_\omega(s_i) + w_\phi (1 - C_\phi)\;,
$$

where the dimensionless mean-centered diagonal advantage is

$$
A_i = \frac{h_i - \bar{h}}{N J_{\text{rms}}}, \qquad h_i = \langle s_i | H_\theta | s_i \rangle, \qquad \bar{h} = \sum_{j=1}^M q_j h_j\;.
$$

#### Proof of Zero-Baseline Property

The expectation of advantage $A_i$ over the Ritz target distribution $q$ is strictly zero.

$$
\langle A \rangle_q = \sum_{i=1}^M q_i A_i = \sum_{i=1}^M q_i \frac{h_i - \bar{h}}{N J_{\text{rms}}} = \frac{1}{N J_{\text{rms}}} \left( \sum_{i=1}^M q_i h_i - \bar{h} \sum_{i=1}^M q_i \right) = 0\;.
$$

Centering removes a constant energy offset but does not guarantee boundedness. The amplitude coefficient is $a_i=q_i(1-w_EA_i)-w_H/M$. Every $a_i$ must be nonnegative for the loss to be bounded below over normalized probabilities. This condition is checked before fitting; both auxiliary weights default to zero. The diagonal advantage omits off-diagonal energy contributions and is not the VMC energy gradient.

### B. Global-Phase-Invariant Phase Coherence ($C_\phi$)

Phase coherence between the NQS phase $\arg \phi_\omega(s_i)$ and the Ritz vector phase $\arg v_{0,i}$ is

$$
C_\phi = \left| \sum_{i=1}^M \widetilde{q}_i e^{i (\arg \phi_\omega(s_i) - \arg v_{0,i})} \right|, \qquad \widetilde{q}_i = \frac{q_i \mathbf{1}_{v_{0,i} \ne 0}}{\sum_{j} q_j \mathbf{1}_{v_{0,j} \ne 0}}\;.
$$

#### Proof of Global Phase Invariance

Under any global phase transformation of the NQS, $\phi_\omega(s) \to e^{i\alpha}\phi_\omega(s)$ with $\alpha \in \mathbb{R}$ a constant phase shift, the coherence becomes

$$
C_\phi' = \left| \sum_{i=1}^M \widetilde{q}_i e^{i (\arg \phi_\omega(s_i) + \alpha - \arg v_{0,i})} \right| = \left| e^{i \alpha} \sum_{i=1}^M \widetilde{q}_i e^{i (\arg \phi_\omega(s_i) - \arg v_{0,i})} \right| = |e^{i \alpha}| \cdot C_\phi = C_\phi\;.
$$

Thus $C_\phi \in [0, 1]$ is strictly invariant under arbitrary global phase gauge choices.

---

## VII. Diagnostics and Failure Criteria

1. **Ritz Error & Variational Bound**: $\epsilon_0 \ge E_{\text{exact}}$.
2. **Epstein-Nesbet PT2 Correction**: $\Delta E_2$ diagnoses weight outside $S_M$. $\epsilon_0 + \Delta E_2$ is nonvariational.
3. **Falsifiable Success Criteria**:
   - $E_{\text{Ritz}}$ decreases at matched $M$.
   - Post-teacher VMC energy decreases at matched sample count.
   - Operator growth $g_{\text{op}}$ remains bounded.

---

## VIII. Limitations

1. **Sample Bias**: Frequency sampling favors visible states; PT2 reaches 1-hop connected states.
2. **Nonvariational PT2**: $\epsilon_0 + \Delta E_2$ can overestimate ground state energy near crossings.
3. **Operator Expansion**: Deep circuits $d \gg 3$ cause exponential term growth $|H_\theta| \propto 3^d$.

---

## References

1. A. Chen and M. Heyl, [arXiv:2302.01941](https://arxiv.org/abs/2302.01941).
2. R. Rende *et al.*, [arXiv:2310.05715](https://arxiv.org/abs/2310.05715).
3. A. Malyshev, M. Schmitt, and A. I. Lvovsky, [arXiv:2408.07625](https://arxiv.org/abs/2408.07625).
4. J.-Y. Chang *et al.*, [arXiv:2606.26760](https://arxiv.org/abs/2606.26760).
