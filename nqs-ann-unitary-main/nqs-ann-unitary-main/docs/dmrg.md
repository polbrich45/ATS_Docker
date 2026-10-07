# Matrix Product States (MPS), Matrix Product Operators (MPO), and DMRG

This document details the theoretical formulation of Matrix Product States (MPS), Matrix Product Operators (MPO), the Density Matrix Renormalization Group (DMRG) algorithm, and the specific physical and computational challenges that arise when applying DMRG to rotated Hamiltonians $H_\theta = U_\theta^\dagger H U_\theta$.

For overall workflow details, see [`theory.md`](theory.md) and [`method.md`](method.md).

---

## 1. Mathematical Formulations: MPS, MPO, and DMRG

### A. Matrix Product State (MPS) Representation
For an $N$-site spin-$1/2$ quantum state $|\Psi\rangle = \sum_{s_1, \dots, s_N} C_{s_1, \dots, s_N} |s_1 \dots s_N\rangle$, an MPS factorizes the rank-$N$ coefficient tensor $C_{s_1, \dots, s_N}$ into a sequence of site-local tensors:

$$
C_{s_1, \dots, s_N} = \sum_{\alpha_1, \dots, \alpha_{N-1}=1}^\chi A_{1,\alpha_1}^{(s_1)} A_{\alpha_1, \alpha_2}^{(s_2)} \dots A_{\alpha_{N-1}}^{(s_N)},
$$

where $s_j \in \{0, 1\}$ is the physical index at site $j$, $\alpha_j \in \{1, \dots, \chi\}$ are virtual bond indices, and $\chi$ is the **MPS bond dimension**.

#### Entanglement Bound
By the Schmidt decomposition across spatial cut $A_k = \{1, \dots, k\}$, the maximum bipartite entanglement entropy $S(A_k)$ representable by an MPS is bounded by the bond dimension:

$$
S(A_k) \le \log_2 \chi.
$$

If a state obeys an entanglement area law $S(A_k) = O(1)$, an MPS with finite $\chi = O(1)$ represents the state with high fidelity. If $S(A_k) = O(L)$, $\chi \propto e^{S}$ must grow exponentially.

### B. Matrix Product Operator (MPO) Representation
A Hamiltonian $H = \sum_{s, s'} W_{s_1, \dots, s_N}^{s_1', \dots, s_N'} |s_1 \dots s_N\rangle \langle s_1' \dots s_N'|$ is represented as an MPO by factorizing its operator matrix elements:

$$
W_{s_1, \dots, s_N}^{s_1', \dots, s_N'} = \sum_{\beta_1, \dots, \beta_{N-1}=1}^{D_{\text{MPO}}} W_{1, \beta_1}^{(s_1, s_1')} W_{\beta_1, \beta_2}^{(s_2, s_2')} \dots W_{\beta_{N-1}}^{(s_N, s_N')},
$$

where $D_{\text{MPO}}$ is the **MPO bond dimension**, measuring the spatial interaction range and non-locality of the operator. Nearest-neighbor 1D Hamiltonians have small $D_{\text{MPO}} = 4 \dots 5$, whereas long-range or multi-site interaction operators have large $D_{\text{MPO}}$.

### C. Density Matrix Renormalization Group (DMRG) Algorithm
DMRG performs iterative local variational optimization of the MPS tensors $A^{(s_j)}$ site-by-site (or block-by-block) to minimize the Rayleigh quotient:

$$
E_{\text{DMRG}} = \min_{A} \frac{\langle \Psi(A) | H_{\text{MPO}} | \Psi(A) \rangle}{\langle \Psi(A) | \Psi(A) \rangle}.
$$

Each sweep solves local effective eigensystems $(W \cdot H_{\text{env}}) A = E A$ and truncates virtual bond dimension $\chi$ via Singular Value Decomposition (SVD) according to a singular value cutoff $\epsilon_{\text{SVD}}$.

---

## 2. What Is Problematic for Adaptive-Basis NQS Rotations ($H_\theta$)?

Applying DMRG as a reference solver on rotated Hamiltonians $H_\theta = U_\theta^\dagger H U_\theta$ introduces three major theoretical and computational challenges:

### A. The Fundamental Trade-off: State Disentangling vs. Operator Expansion

$$\text{Unitary Rotation } U_\theta \implies \begin{cases} \text{State Entanglement } S(\Psi) \downarrow & \implies \text{Required MPS Bond } \chi \downarrow \text{ (Good)}, \\ \text{Operator Growth } |H_\theta| \uparrow & \implies \text{MPO Bond Dimension } D_{\text{MPO}} \uparrow \text{ (Problematic)}. \end{cases}$$

1. **State Disentangling (Good)**: A shallow circuit $U_\theta$ rotates the computational basis, disentangling local spin correlations and lowering physical entanglement entropy $S(\Psi)$. This allows an MPS with a smaller bond dimension $\chi$ to represent the rotated state $|\phi_\omega\rangle$.
2. **Operator Expansion (Problematic)**: Conjugation $H_\theta = U_\theta^\dagger H U_\theta$ transforms local 2-site terms into long-range multi-site Pauli strings. The number of terms $|H_\theta|$ expands rapidly, driving the MPO bond dimension $D_{\text{MPO}}$ from $D_{\text{MPO}} \approx 5$ to $D_{\text{MPO}} \approx 50 \dots 200+$.
3. **Net Computational Cost**: DMRG sweep time scales as $O(N \cdot D_{\text{MPO}} \cdot \chi^3 + N \cdot D_{\text{MPO}}^2 \cdot \chi^2)$. If $D_{\text{MPO}}$ increases significantly, the computational cost of DMRG on $H_\theta$ can exceed that on the unrotated Hamiltonian $H$, even though state entanglement $S(\Psi)$ decreased!

### B. 1D Mapping of 2D rotated Hamiltonians
- Mapping a 2D spin lattice (e.g. $4 \times 4$ or $6 \times 6$ square $J_1$-$J_2$) onto a 1D MPS chain introduces long-range interaction bonds spanning distance $L_x$.
- When $U_\theta$ adds additional 2-site or 3-site gates (such as `string_givens`), interaction ranges extend further across the 1D snake ordering, causing $D_{\text{MPO}}$ to grow exponentially with transverse system width $L_y$.

### C. Nonvariational MPO Compression Errors
- If $D_{\text{MPO}}$ becomes too large, MPO compression algorithms apply a truncation cutoff (`dmrg_mpo_cutoff=1e-10` or `dmrg_mpo_max_bond`).
- Truncating MPO tensors replaces the exact rotated Hamiltonian $H_\theta$ with an approximate operator $\widetilde{H}_\theta$.
- **Problem**: DMRG eigenvalues evaluated on $\widetilde{H}_\theta$ are **no longer variational upper bounds** on the true ground state energy $E_0$ of $H$. MPO truncation errors can introduce non-physical energy shifts, making comparisons between $H$ and $H_\theta$ non-rigorous.

---

## 3. Configuration & Diagnostic Recommendations

To evaluate DMRG fairly on $H$ vs. $H_{\theta^*}$:

```python
FinalConfig(
    solver="dmrg",
    ritz_size=0,
    dmrg_mps_bond_dim=128,  # Match max MPS bond dimension
    dmrg_mpo_max_bond=0,    # 0 = Uncapped MPO bond dimension (prevents MPO truncation error)
    dmrg_mpo_cutoff=1e-10,
    dmrg_tol=1e-8,
    dmrg_sweeps=10,
)
```

1. **Keep MPO Bond Uncapped (`dmrg_mpo_max_bond=0`)**: Prevents non-variational MPO compression errors.
2. **Track Dual Metrics**: Record both final MPS bond dimension $\chi$ and MPO bond dimension $D_{\text{MPO}}$ alongside energy.
3. **Validate Spectrum on Small Systems**: For $N \le 12$, verify that exact diagonalization (ED) of $H_{\theta^*}$ matches $H$ to machine precision before interpreting DMRG results.
