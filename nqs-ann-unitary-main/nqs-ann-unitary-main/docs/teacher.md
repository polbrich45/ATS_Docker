# Ritz-to-NQS Teacher (State Transfer)

This document defines the Ritz-to-NQS teacher in [`src/teacher.py`](../src/teacher.py), inspired by the sample-diagonalize-distill structure of [HI-NQS](https://arxiv.org/abs/2606.26760). The default spin teacher uses joint probabilities and relative phases. For the complete objectives, see [`theory.md`](theory.md) and [`loss.md`](loss.md).

---

## 1. Input from the Sampled Ritz Problem

At a fixed circuit angle $\theta$, the NQS represents a variational state for the rotated Hamiltonian $H_\theta = U_\theta^\dagger H U_\theta$.

For the retained computational basis $S_M = \{s_1, \dots, s_M\}$ and lowest Ritz eigenpair $(E_{\text{Ritz}}, v_0)$, the teacher receives:
- Target probabilities $q_i = |v_{0,i}|^2 / \|v_0\|^2$ over $s_i \in S_M$.
- Rotated diagonal elements $h_i = \langle s_i | H_\theta | s_i \rangle$.
- Mean-centered dimensionless diagonal advantage:
  $$
  A_i = \frac{h_i - \bar{h}}{N J_{\text{rms}}}, \qquad \bar{h} = \sum_{j=1}^M q_j h_j.
  $$

---

## 2. Theoretical Derivation of Loss Components

The complete teacher loss functional is:

$$
\mathcal{L}_T = \frac{1}{N} D_{\text{KL}}(q \parallel p_\omega) + \frac{w_E}{N} \sum_{i=1}^M q_i A_i \log p_\omega(s_i) + \frac{w_H}{M N} \sum_{i=1}^M \log p_\omega(s_i) + w_\phi (1 - C_\phi).
$$

### A. Probability Distillation ($D_{\text{KL}}$)
The KL divergence per site between the Ritz vector Born distribution $q$ and NQS distribution $p_\omega$ is:

$$
\mathcal{L}_{\text{KL}} = \frac{1}{N} D_{\text{KL}}(q \parallel p_\omega) = \frac{1}{N} \sum_{i=1}^M q_i \log \frac{q_i}{p_\omega(s_i)} = \frac{-1}{N} \sum_{i=1}^M q_i \log p_\omega(s_i) - \frac{H(q)}{N}.
$$

Because $H(q)$ is parameter-independent, $\nabla_\omega \mathcal{L}_{\text{KL}} = \frac{-1}{N} \sum q_i \nabla_\omega \log p_\omega(s_i)$. $\mathcal{L}_{\text{KL}} \ge 0$ reaches zero if and only if $p_\omega(s_i) = q_i$ on $S_M$ and $p_\omega(s) = 0$ for $s \notin S_M$.

Define the retained mass $m=\sum_{i\in S_M}p_\omega(s_i)$ and conditional probabilities $p_\omega(s_i\mid S_M)=p_\omega(s_i)/m$. The joint KL separates into

$$
D_{\rm KL}(q\Vert p_\omega)=D_{\rm KL}(q\Vert p_\omega(\cdot\mid S_M))-\log m\;.
$$

The implementation reports `conditional_kl_per_site`, `mass_loss_per_site`, and `subspace_mass` from the same network evaluation used by the loss. Their first two values sum to `teacher_kl_per_site`. The weighted `phase_loss` is dimensionless and is not divided by $N$. Fitting a truncated Ritz target can reduce the KL while raising physical energy through loss of amplitudes outside the retained space. Physical energies before and after teaching and recovery therefore remain separate diagnostics; they determine proposal acceptance rather than the teacher objective.

The spin loop retains the state before recovery if the measured energy increase exceeds `acceptance_sigma` times the combined standard error, plus the numerical tolerance. `recovery_restored_start` records this choice. The recovery endpoint remains in the history and its full cost remains counted; restoring the earlier state needs no new evaluation. With teaching disabled, this checkpoint is the unchanged NQS in the proposed basis.

It also reports `teacher_fidelity`, the squared full-state overlap with the normalized Ritz target,

$$
F_T=\left|\sum_{i=1}^{M}\frac{v_{0,i}^*}{\|v_0\|}\psi_\omega(s_i)\right|^2\;.
$$

The autoregressive NQS is normalized over its full physical sector; its sampled amplitudes are not renormalized in this expression. Thus $F_T$ includes missing retained mass as well as amplitude and phase errors. Phase coherence alone does not measure state fidelity. This diagnostic uses the existing teacher forward pass, without ground-state information or extra samples. During small-system teacher energy probes, the basis states and fixed sparse $H_\theta$ are cached; full reference and entropy diagnostics are evaluated at stage endpoints.

### B. Mean-Centered REINFORCE Advantage ($w_E$)
The advantage term rewards configurations with below-average diagonal energy ($h_i < \bar{h} \implies A_i < 0$) by increasing their probability $p_\omega(s_i)$, while penalizing configurations with above-average diagonal energy ($h_i > \bar{h} \implies A_i > 0$).

#### Proof of Zero Expected Baseline
$$
\langle A \rangle_q = \sum_{i=1}^M q_i A_i = \sum_{i=1}^M q_i \frac{h_i - \bar{h}}{N J_{\text{rms}}} = \frac{\sum q_i h_i - \bar{h} \sum q_i}{N J_{\text{rms}}} = 0.
$$
Centering removes a constant diagonal-energy offset; it does not guarantee boundedness. Define the combined amplitude coefficient $a_i = q_i(1-w_E A_i)-w_H/M$. The amplitude loss is $-\sum_i a_i\log p_\omega(s_i)/N$ plus a constant. It is bounded below over normalized probabilities if every $a_i\geq0$. A negative coefficient permits divergence to $-\infty$ as the corresponding probability vanishes. The implementation checks this condition before teaching. Both auxiliary weights default to zero. The diagonal bias is a heuristic, not the physical energy gradient, because it omits off-diagonal matrix elements.

### C. Anti-Concentration Regularization ($w_H$)
The uniform log-probability term $\mathcal{L}_H = \sum_i\log p_\omega(s_i)/(MN)$ is not Shannon entropy or a physical entropy. For positive $w_H$, it rewards making some sampled probabilities small. It does not guarantee increased probability outside the subspace. If any $q_i=0$, every positive $w_H$ violates the boundedness condition. Leave this term disabled for the reference experiment.

### D. Global-Phase-Invariant Phase Coherence ($w_\phi$)
Phase coherence between NQS phase $\phi_\omega(s_i) = \operatorname{Im} \log \psi_\omega(s_i)$ and Ritz phase $\phi_i^R = \arg v_{0,i}$ is:

$$
C_\phi = \left| \sum_{i=1}^M \widetilde{q}_i e^{i (\phi_\omega(s_i) - \phi_i^R)} \right|, \qquad \mathcal{L}_\phi = 1 - C_\phi.
$$

#### Proof of Gauge Invariance
Under global phase transformation $\phi_\omega(s) \to \phi_\omega(s) + \alpha$:

$$
C_\phi' = \left| e^{i \alpha} \sum_{i=1}^M \widetilde{q}_i e^{i (\phi_\omega(s_i) - \phi_i^R)} \right| = |e^{i \alpha}| \cdot C_\phi = C_\phi.
$$
$C_\phi = 1$ when all relative phases match, while remaining completely invariant under arbitrary global phase gauge choices.

Gauge invariance does not ensure a useful gradient everywhere. For a real signed target and constant NQS phase, every phase difference is either zero or $\pi$ after choosing a global gauge. If the signed target weights sum to a nonzero value, all derivatives of $1-C_\phi$ with respect to the individual NQS phases vanish, even when signs are wrong. For target weights $(0.8,0.2)$ and signs $(+,-)$, this stationary loss is $0.4$. A positive Ising NQS can approach this situation before rotation. Phase coherence, teacher fidelity, and post-teacher physical energy must therefore be checked separately; a stalled phase loss does not establish successful sign learning.

---

## 3. Hyperparameter Configuration

| `TeacherConfig` Field | Mathematical Term | Role | Default |
| :--- | :--- | :--- | :--- |
| `n_teacher` | - | Teacher update steps | `500` (CLI: `50`) |
| `optimizer` | - | Adam; optional experimental fixed-target L-BFGS | `adam` |
| `learning_rate` | - | Optimizer step size | `1e-3` |
| `grad_clip` | - | Global gradient norm clip | `1.0` |
| `reinforce_weight` | $w_E$ | Optional diagonal-energy bias | `0.0` |
| `entropy_weight` | $w_H$ | Optional uniform log-probability term | `0.0` |
| `phase_weight` | $w_\phi$ | Global-phase-invariant phase weight | `0.1` |
| `nqs_energy_every` | - | Physical NQS energy observation every this many completed spin-teacher updates; zero retains endpoints | `0` |

The spin adaptive loop records these observations in `teacher_nqs_history`. Angles remain fixed while NQS parameters change. Loss history entries precede updates; energy probes and the final teacher summary follow completed updates. Probes are diagnostic and their cost is included in evaluation time. Neither small-system exact probes nor larger-system Monte Carlo probes advance the training random stream: the exact path never touches it, and the Monte Carlo path draws from a key folded off the current one, so enabling `nqs_energy_every` leaves the sampling trajectory unchanged.

L-BFGS requires `grad_clip=0` to preserve the line-search gradient. Its cached gradient is reused only within one fixed-target fit; initialization computes value and gradient together. `function_evaluations` counts the update evaluations, line-search trials, and final evaluation. The Ising audits in `validation.md` do not establish an advantage for this option, so Adam remains the default.
