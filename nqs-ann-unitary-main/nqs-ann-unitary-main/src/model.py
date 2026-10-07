r"""Pre-LN autoregressive Transformer with normalized Born amplitudes and phase.

For a spin configuration ``s``, the model represents

``log psi(s) = 1/2 sum_j log p(s_j | s_<j) + i sum_j phi_j(s_j | s_<j)``.

The exact conditional normalization supports independent Born sampling; the
phase head carries sign/phase structure without changing those probabilities.
"""

from __future__ import annotations
import math
from typing		import Any, Union

import numpy as np

try:
    import	flax.linen as nn
    import	jax
    import	jax.numpy as jnp
    from	netket.models import AbstractARNN
except ImportError as e:
    raise ImportError("This module requires JAX, Flax and NetKet. Install them with `pip install jax jaxlib flax netket`") from e

# ------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Pre-normalized causal Transformer block.

    This layer applies Multi-Head Self-Attention followed by a Feed-Forward network (MLP),
    using Layer Normalization prior to each block (Pre-LN) and residual connections.
    Causal masking is applied in self-attention to preserve the autoregressive property,
    ensuring site j only receives information from sites 1 to j-1.

    Attributes
    ----------
    width (int)
        The hidden feature dimension (number of channels).
    heads (int)
        Number of attention heads (must divide width).
    mlp_ratio (int)
        Ratio of the MLP hidden layer size to the width.
    """

    width		: int			# hidden dimension
    heads		: int			# attention heads (must divide width)
    mlp_ratio	: int			# ratio of MLP hidden dimension to width
    param_dtype : Any = None # None dynamically resolves to float64/float32 based on JAX x64 config

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None, *, decode: bool = False) -> jax.Array:
        """
        Apply the Transformer block to the input sequence.

        Parameters
        ----------
        x (jax.Array)
            Input tensor of shape (batch_size, n_sites, width).
        mask (jax.Array)
            Causal attention mask of shape (1, n_sites, n_sites) or similar.

        Returns
        -------
        jax.Array
            Processed tensor of shape (batch_size, n_sites, width).
        """
        dtype = self.param_dtype
        if dtype is None:
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        # 1. Pre-Attention normalization
        y = nn.LayerNorm(param_dtype=dtype)(x)

        # 2. Causal Multi-Head Self-Attention
        # We omit bias to respect spin symmetries and simplify parameters.
        # deterministic=True disables dropout for reproducible NQS variational optimization.
        y = nn.MultiHeadDotProductAttention(
            num_heads		= self.heads,	# number of attention heads
            qkv_features	= self.width,	# hidden dimension (must be divisible by heads)
            out_features	= self.width,	# output dimension (must be divisible by heads)
            use_bias		= False,		# no bias for simplicity and potential symmetry benefits
            deterministic	= True,			# disable dropout for simplicity and potential symmetry benefits
            param_dtype		= dtype,
            decode			= decode,		# enable caching for autoregressive decoding
        )(y, mask=mask)

        # 3. First residual connection: x_mid = x + Attention(LN(x))
        x = x + y

        # 4. Pre-MLP normalization
        y = nn.LayerNorm(param_dtype=dtype)(x)

        # 5. Position-wise Feed-Forward network (MLP)
        # Expands the representation and applies the GeLU non-linear activation.
        y = nn.Dense(self.mlp_ratio * self.width, param_dtype=dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, param_dtype=dtype)(y)

        # 6. Second residual connection: x_mid + MLP(LN(x_mid))
        return x + y


class ARTransformer(AbstractARNN):
    r"""
    Normalized amplitude autoregression with a learned complex phase.

    Physically, this represents a Neural Quantum State (NQS) where:
      - Log-probability amplitudes log |psi(S)| are computed from the amplitude head logits.
      - Complex phases theta(S) are accumulated from site-by-site phase updates.

    This model embeds spin configuration values and uses causal masking to enforce the
    autoregressive property: P(S_j | S_{<j}) and phi_j(S_j | S_{<j}).

    It subclasses :class:`netket.models.AbstractARNN`, so NetKet's
    :class:`netket.sampler.ARDirectSampler`, :class:`netket.vqs.MCState`, and the
    VMC drivers/optimizers work directly on it.	 The NetKet
    contract is implemented through :meth:`conditionals_log_psi`; the inherited
    ``__call__`` / ``conditionals`` are derived from it; ``conditional`` uses cached decoding. VMC and
    support diagnostics use NetKet's with-replacement sampler; compact Ritz seeds use the unique
    conditioned-Gumbel sampler in :mod:`src.nqs`.

    Convention
    ----------
    NetKet ``Spin(s=1/2)`` has ``local_states = [+1, -1]`` with local indices
    ``[0, 1]``; ``operators.py`` encodes Pauli strings with bit ``0 <-> +1`` as
    well, so the model channel index *is* the NetKet local index (no relabelling).
    ``conditionals_log_psi`` therefore receives physical spins ``{+1, -1}`` and maps
    them to bits with ``b = (1 - sigma) / 2 = hilbert.states_to_local_indices``.

    Parameters
    ----------
    hilbert (netket.hilbert.Spin)
    Homogeneous spin or spin-orbital fermion Hilbert space
    (``hilbert.size == n_sites``). Pass an unconstrained space; the conditional
    masks enforce the requested spin or particle sector.
    n_sites (int)
        Number of lattice sites (sequence length). Determines the input and output dimensions, input
        tokenization, and positional encodings.
    width (int)
        Hidden dimension size for embeddings and block layers.
    depth (int)
        Number of Transformer blocks.
    heads (int)
        Number of attention heads (must divide width). Heads allow the model to learn multiple distinct patterns of correlation.
    mlp_ratio (int)
        Expansion ratio for the MLP hidden layers.
    phase_head_depth (int)
        Residual MLP depth used only by the phase/sign head.
    n_up (int, optional)
        Fixed number of up spins. Restricts states to a magnetization sector
        with S^z_total = (2 * n_up - n_sites) / 2. If ``None``, the model uses
        the full Hilbert space.
    particle_counts (tuple[int, int])
        Fixed occupancies in NetKet's consecutive up/down orbital blocks.
    local_occupations (bool)
        Interpret local states as fermionic occupations ``{0, 1}``.
    reference_phase (tuple[float, ...])
        Optional reference phase vector (e.g. for Marshall sign).
    machine_pow (int)
        Exponent normalizing ``|psi|`` -> ``|psi|^machine_pow`` in the NetKet
        ``conditionals`` (``2`` recovers the Born probabilities; required by
        :class:`AbstractARNN`).
    """

    n_sites				: int						# number of lattice sites
    width				: int						# hidden dimension
    depth				: int						# number of Transformer blocks
    heads				: int						# attention heads (must divide width)
    mlp_ratio			: int						# ratio of MLP hidden dimension to width

    # Phase head parameters: residual MLP for phase/sign expressivity
    phase_head_depth	: int = 1					# residual MLP blocks only for phase/sign expressivity
    phase_head_width	: int = 0					# hidden width for phase head; 0 -> width
    phase_scale			: float = 1.0				# learned phase increment multiplier
    phase_init_std		: float = 0.05				# small learned phase perturbation

    # Optional constraints and symmetries
    n_up				: Union[int, None] = None	# fixed up-spin count; None uses the full Hilbert space
    particle_counts		: tuple[int, int] = ()		# occupied orbitals in the two spin blocks, for fermions
    local_occupations	: bool = False				# fermion local states are {0, 1}, not spin {+1, -1}
    reference_phase		: tuple[float, ...] = ()	# optional fixed reference phase for each site (e.g. for the Marshall sign)
    machine_pow			: int = 2					# |psi|^machine_pow normalization for NetKet conditionals (2 = Born rule)
    param_dtype			: Any = None				# None dynamically resolves to float64/float32 based on JAX x64 config
    num_embeddings		: int = 3					# local-index tokens (0: up/+1, 1: down/-1, 2: start)

    def setup(self) -> None:
        r"""
        Initialize embedding, positional encodings, blocks, and output heads.

        Ansatz is:
        1. Embedding layer maps input tokens to hidden features.
        2. Positional encodings are added to the embeddings to provide site-specific information.
        3. A stack of causal Transformer blocks processes the sequence.
        4. Final layer normalization is applied.
        5. Two output heads predict amplitude logits and phase increments for each site.

        Mathematically, the model computes:
        - Amplitude logits: logits_j(S_{<j}) for each site j,
        - Phase increments: phi_j(S_{<j}) for each site j,
        which are combined to yield the log-wavefunction:
        ln psi(S) = 1/2 \sum_j ln P(S_j | S_{<j}) + i \sum_j phi_j(S_j | S_{<j})
        """
        dtype = self.param_dtype
        if dtype is None:
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        # Embedding tokens follow NetKet local indices: 0=up/+1, 1=down/-1,
        # with 2 reserved for the autoregressive start token.
        # start token means "no previous spins" and allows the first site to have a valid conditional distribution.
        self.embedding		= nn.Embed(num_embeddings=self.num_embeddings, features=self.width, param_dtype=dtype)

        # Positional embedding: learns spatial site-specific features (essential for translation and geometry)
        self.position		= self.param("position", nn.initializers.normal(stddev=0.02), (self.n_sites, self.width), dtype)

        # Stack of causal Transformer blocks
        self.blocks			= [TransformerBlock(self.width, self.heads, self.mlp_ratio, param_dtype=dtype) for _ in range(self.depth)]

        # Final layer normalization applied to the combined outputs of the stack
        self.norm			= nn.LayerNorm(param_dtype=dtype)

        # Start from the uniform Born distribution (within the exact sector mask).
        # A default dense projection can put almost all initial mass on one state,
        # leaving a finite-sample SR step blind before training has begun.
        self.amplitude_head = nn.Dense(
            2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            param_dtype=dtype,
        )

        phase_width			= self.width if self.phase_head_width <= 0 else self.phase_head_width
        self.phase_norms	= [nn.LayerNorm(param_dtype=dtype) for _ in range(self.phase_head_depth)]
        self.phase_in		= [nn.Dense(phase_width, param_dtype=dtype) for _ in range(self.phase_head_depth)]
        self.phase_out		= [nn.Dense(self.width, param_dtype=dtype) for _ in range(self.phase_head_depth)]

        # A small random phase breaks the stationary positive-state phase manifold
        # for sign-changing real Hamiltonians. Stoquastic Ising needs no such sign change.
        self.phase_head     = nn.Dense(2, kernel_init=nn.initializers.normal(stddev=self.phase_init_std), bias_init=nn.initializers.zeros, param_dtype=dtype)

    def _heads(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Return conditional amplitude logits and phase increments for all sites.

        Low-level head used by the NetKet contract (``conditionals_log_psi``).	It is
        *not* the NetKet ``conditionals`` (which returns normalized Born probabilities).

        Parameters
        ----------
        tokens (jax.Array)
            Causal shifted spin *bit* tokens of shape (batch_size, n_sites) in
            ``{0, 1, 2}`` (2 = start token).

        Returns
        -------
        amplitude_logits (jax.Array)
            Logits of shape (batch_size, n_sites, 2).
        phase_increments (jax.Array)
            Phase updates of shape (batch_size, n_sites, 2).
        """
        h = self._hidden(tokens)
        if self.phase_scale == 0.0:
            logits = self.amplitude_head(h)
            return logits, jnp.zeros_like(logits)
        phase_h = h

        # 5. Phase head: residual MLP for phase expressivity
        for norm, dense_in, dense_out in zip(self.phase_norms, self.phase_in, self.phase_out):
            y		= norm(phase_h)
            y		= dense_in(y)
            y		= nn.gelu(y)
            y		= dense_out(y)
            phase_h = phase_h + y # residual connection
        return self.amplitude_head(h), self.phase_scale * self.phase_head(phase_h)

    def _hidden(self, tokens: jax.Array, *, decode: bool = False, index=None) -> jax.Array:
        """Shared full-sequence and cached one-token Transformer evaluation."""
        position = self.position[None, :, :] if index is None else self.position[index][None, None, :]
        h = self.embedding(tokens) + position
        mask = None if decode and index is not None else nn.make_causal_mask(jnp.ones((1, tokens.shape[1]), dtype=bool))
        for block in self.blocks:
            h = block(h, mask, decode=decode)
        return self.norm(h)

    @nn.compact
    def conditional(self, inputs: jax.Array, index: int) -> jax.Array:
        """Born conditional with fresh per-draw key/value caches in ARDirectSampler."""
        if not self.is_mutable_collection("cache"):
            return self.conditionals(inputs)[:, index, :]
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        bits = inputs_to_bits(self, inputs).astype(jnp.int32)
        initialized = self.has_variable("cache", "initialized")
        self.variable("cache", "initialized", lambda: jnp.array(True))
        if not initialized:
            # NetKet initializes caches on the full sequence before its sampling scan.
            start = jnp.full((bits.shape[0], 1), 2, dtype=jnp.int32)
            tokens = jnp.concatenate((start, bits[:, :-1]), axis=1)
            logits = self.amplitude_head(self._hidden(tokens, decode=True))
            return nn.softmax(self._sector_mask(logits, bits)[:, index, :], axis=-1)
        previous = jnp.take(bits, jnp.maximum(index - 1, 0), axis=1)
        tokens = jnp.where(index == 0, 2, previous)[:, None]
        logits = self.amplitude_head(self._hidden(tokens, decode=True, index=index))
        # Reuse the exact sector prior; future bits never enter the selected prefix mask.
        logits = jnp.broadcast_to(logits, (bits.shape[0], self.n_sites, 2))
        return nn.softmax(self._sector_mask(logits, bits)[:, index, :], axis=-1)

    def _sector_mask(self, logits: jax.Array, bits: jax.Array) -> jax.Array:
        """
        Apply a dynamic mask to amplitude logits to strictly enforce fixed-count sectors.

        Each block of :func:`sector_blocks` fixes its number of 1 bits: down spins
        for a magnetization sector, occupied orbitals for each fermion spin block.
        This method tracks choices made up to each site j and masks out choices
        that would violate the constraint by setting their logits to -inf.

        Parameters
        ----------
        logits (jax.Array)
            Logits of shape (batch_size, n_sites, 2).
        bits (jax.Array)
            The chosen spin values in {0, 1} of shape (batch_size, n_sites).

        Returns
        -------
        jax.Array
            Masked logits of shape (batch_size, n_sites, 2).
        """
        blocks = sector_blocks(self)
        if not blocks:
            return logits
        masked = []
        for start, stop, n_ones in blocks:
            block_bits		= bits[:, start:stop]
            ones_before		= jnp.cumsum(block_bits, axis=1) - block_bits
            zeros_before	= jnp.arange(stop - start)[None, :] - ones_before
            remaining		= jnp.stack((stop - start - n_ones - zeros_before, n_ones - ones_before), axis=-1)
            # The remaining-count factor is the exact hypergeometric base measure.
            # With zero learned logits it assigns equal probability to every complete
            # configuration in the fixed sector, rather than favoring bits chosen early.
            masked.append(jnp.where(remaining > 0, logits[:, start:stop] + jnp.log(jnp.maximum(remaining, 1)), -jnp.inf))
        return jnp.concatenate(masked, axis=1)

    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        r"""
        Per-site conditional log-wavefunctions (the NetKet ``AbstractARNN`` contract).

        Returns ``log_psi`` of shape ``(batch, n_sites, 2)`` such that choosing local
        value ``b`` at site ``j`` contributes

            log_psi[:, j, b] = 1/2 ln P(b | S_{<j}) + i phi_j(b | S_{<j})

        so that the inherited ``__call__`` (gather over the realized indices + sum)
        yields ``ln psi(S) = 1/2 sum_j ln P + i sum_j phi_j``, the complex
        log-wavefunction, and the inherited ``conditionals`` (with ``machine_pow=2``)
        returns the Born probabilities ``P(b | S_{<j})``.

        Args:
            inputs (jax.Array): Physical spin configurations of shape
                (batch, n_sites) with values in ``{+1, -1}`` (NetKet convention).

        Returns:
            jax.Array: Complex ``log_psi`` of shape (batch, n_sites, 2).
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        # Fast JAX bit mapping: +1 -> 0, -1 -> 1 (matches NetKet local indices for Spin 1/2)
        bits	= inputs_to_bits(self, inputs).astype(jnp.int32)

        # Shift inputs to build causal tokens: tokens = [StartToken, S_1, ..., S_{N-1}]
        start	= jnp.full((bits.shape[0], 1), 2, dtype=jnp.int32)
        tokens	= jnp.concatenate((start, bits[:, :-1]), axis=1)

        # Amplitude logits and phase increments, then enforce magnetization conservation.
        logits, phases	= self._heads(tokens)
        logits			= self._sector_mask(logits, bits)

        # Born-normalized amplitude (1/2 ln P) plus the learned phase, per site, per channel.
        log_prob	= nn.log_softmax(logits, axis=-1)			# (batch, n_sites, 2)
        # A positive state needs a real log-wavefunction, so SR uses the smaller real kernel.
        log_psi		= 0.5 * log_prob if self.phase_scale == 0.0 and not self.reference_phase else 0.5 * log_prob + 1.0j * phases

        # Optional fixed reference (Marshall) phase: choosing bit b at site j adds
        # b * pi * ref_j, i.e. channel 1 gets +pi ref_j and channel 0 gets 0.
        if self.reference_phase:
            ref		= jnp.asarray(self.reference_phase, dtype=log_prob.dtype)	# (n_sites,)
            add		= jnp.stack((jnp.zeros_like(ref), jnp.pi * ref), axis=-1)	# (n_sites, 2)
            log_psi = log_psi + 1.0j * add[None, :, :]

        return log_psi


# ------------------------------------------------------------------------------
# Local encoding and fixed-count sectors. Bits are NetKet local indices:
# spin 0=up(+1), 1=down(-1); fermion 0=empty, 1=occupied.
# ------------------------------------------------------------------------------


def bits_to_inputs(model: Any, bits: Any) -> Any:
    """Map local-index bits to model inputs: occupations unchanged, spins ``1 - 2 b``."""
    return bits if model.local_occupations else 1 - 2 * bits


def inputs_to_bits(model: Any, inputs: Any) -> Any:
    """Inverse of :func:`bits_to_inputs`."""
    return inputs if model.local_occupations else (1 - inputs) // 2


def sector_blocks(model: Any) -> tuple[tuple[int, int, int], ...]:
    """Fixed-count blocks ``(start, stop, n_ones)`` of the autoregressive support.

    A magnetization sector fixes ``n_sites - n_up`` down spins on all sites.
    NetKet ``SpinOrbitalFermions`` stores the up and down orbitals as two
    consecutive blocks, each with its own particle count.
    """
    if model.particle_counts:
        block = model.n_sites // 2
        return tuple((spin * block, (spin + 1) * block, int(count)) for spin, count in enumerate(model.particle_counts))
    return () if model.n_up is None else ((0, model.n_sites, model.n_sites - int(model.n_up)),)


def support_size(model: Any) -> int:
    """Number of bit strings allowed by :func:`sector_blocks`."""
    blocks = sector_blocks(model)
    free = model.n_sites - sum(stop - start for start, stop, _ in blocks)
    return 2 ** free * math.prod(math.comb(stop - start, n_ones) for start, stop, n_ones in blocks)


def in_sector(model: Any, bits: np.ndarray) -> bool:
    """True when every row of ``bits`` satisfies all fixed block counts."""
    bits = np.asarray(bits)
    return all(bool(np.all(bits[:, start:stop].sum(axis=1) == n_ones)) for start, stop, n_ones in sector_blocks(model))


# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
