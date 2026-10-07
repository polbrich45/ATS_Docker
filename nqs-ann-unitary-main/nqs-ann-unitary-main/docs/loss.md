# Losses, Objectives, and Scale Invariance

This document provides the theoretical derivations, scaling normalizations, and mathematical contracts for the rotation and teacher loss functions. For overall formulation, see [`theory.md`](theory.md).

---

## 1. Physical Scale Invariance & $J_{\text{rms}}$ Normalization

Consider a physical Hamiltonian $H = \sum_{P \in \mathcal{P}_N} c_P P$. Under a positive scaling and identity shift:

$$
H \longrightarrow c H + d I, \qquad c > 0, \quad d \in \mathbb{R}.
$$

To ensure that parameter gradient updates $\nabla_\theta \mathcal{L}_\theta$ and $\nabla_\omega \mathcal{L}_T$ are invariant under such unphysical transformations, all energy differences are divided by the root-mean-square Hamiltonian coupling scale:

$$
J_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{P \ne I} |c_P|^2}.
$$

#### Proof of Scale Invariance
Under $H \to c H + d I$, $c_P \to c \, c_P$ for $P \ne I$, so $J_{\text{rms}} \to c \, J_{\text{rms}}$.
The normalized Ritz energy scales as:

$$
\bar{E}' = \frac{\epsilon_0' - d}{N J_{\text{rms}}'} = \frac{(c \epsilon_0 + d) - d}{N (c J_{\text{rms}})} = \frac{c \epsilon_0}{c N J_{\text{rms}}} = \bar{E}.
$$

The implemented loss uses $\epsilon_0/(N J_{\text{rms}})$ without subtracting the identity shift. Its value therefore changes by an additive constant under $H\to cH+dI$, but its angle gradient is unchanged. Positive rescaling leaves both value and gradient unchanged.

---

## 2. Differentiable Rotation Loss Objective

The complete rotation loss functional optimized over shallow circuit angles $\theta$ is:

$$
\mathcal{L}_\theta = \frac{\epsilon_0}{N J_{\text{rms}}} + \lambda_{\text{part}} \bar{S}_{\text{part}} + \lambda_{\text{ent}} \bar{S}_2 + \lambda_{\text{sign}} g_{\text{sign}} + \lambda_{\text{nested}} L_{\text{nested}}.
$$

### Mathematical Formulation of Terms:

1. **Normalized Ritz Energy ($\bar{E}$)**:
   $$\bar{E} = \frac{\epsilon_0}{N J_{\text{rms}}}.$$
   Rayleigh-Ritz variational ground state energy of $H_{S_M}(\theta)$ divided by system size $N$ and $J_{\text{rms}}$.

2. **Nested Ritz Penalty ($L_{\text{nested}}$)**:
   $$L_{\text{nested}} = \frac{\epsilon_{M/2} - \epsilon_M}{N J_{\text{rms}}}\;.$$
   Measures the energy truncation gap between the full subspace $S_M$ and its top half $S_{M/2} = \text{Top}_{M/2} \{|v_{0,s}|^2\}$. The smaller subset is frozen during each rotation block. For $0\leq\lambda_{\text{nested}}\leq1$, the energy terms equal $[(1-\lambda_{\text{nested}})\epsilon_M+\lambda_{\text{nested}}\epsilon_{M/2}]/(N J_{\text{rms}})$. This avoids a negative coefficient on the full-space energy. It encourages compactness but does not guarantee a better NQS.

3. **Subspace Participation Entropy ($\bar{S}_{\text{part}}$)**:
   $$\bar{S}_{\text{part}} = \frac{-\log \sum_{i=1}^M p_i^2}{\log M}, \qquad p_i = \frac{|v_{0,i}|^2}{\|v_0\|^2}.$$
   Normalized Inverse Participation Ratio (IPR) entropy inside $S_M$. $\bar{S}_{\text{part}} \in [0, 1]$.

4. **Projected-State Rényi-2 Entanglement ($\bar{S}_2$)**:
   $$\bar{S}_2 = \frac{\sum_{k=1}^{N-1} -\log \operatorname{Tr} \rho_{A_k}^2}{\sum_{k=1}^{N-1} \log \min(M, 2^{\min(k, N-k)})}.$$
   Sum of ordered-cut Rényi-2 entanglement entropies evaluated on the zero-extended sampled state $|\psi_S\rangle$.

5. **Sampled Nonstoquasticity Indicator ($g_{\text{sign}}$)**:
   $$g_{\text{sign}} = \frac{\sum_{i \ne j} \left[ \max(\operatorname{Re}(H_{S_M})_{ij}, 0)^2 + \operatorname{Im}(H_{S_M})_{ij}^2 \right]}{\sum_{i \ne j} |(H_{S_M})_{ij}|^2}.$$

---

## 3. Ritz-to-NQS Teacher Objective

The state transfer loss $\mathcal{L}_T$ fits NQS parameters $\omega$ to the Ritz ground vector $v_0$:

$$
\mathcal{L}_T = \frac{1}{N} D_{\text{KL}}(q \parallel p_\omega) + \frac{w_E}{N} \sum_{i=1}^M q_i A_i \log p_\omega(s_i) + \frac{w_H}{M N} \sum_{i=1}^M \log p_\omega(s_i) + w_\phi (1 - C_\phi),
$$

where mean-centered advantage is $A_i = \frac{h_i - \bar{h}}{N J_{\text{rms}}}$ with $\bar{h} = \sum_{j=1}^M q_j h_j$, and phase coherence is $C_\phi = \left| \sum_{i=1}^M \widetilde{q}_i e^{i (\arg \phi_\omega(s_i) - \arg v_{0,i})} \right|$.

The default is $w_E=w_H=0$. Optional auxiliary terms require $q_i(1-w_EA_i)-w_H/M\geq0$ for every retained state; centering alone does not ensure boundedness. Neither the sampled entropy nor the diagonal bias certifies improvement in the physical NQS energy. A fixed-angle VMC recovery and energy acceptance check follow teaching.
