# copied from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
import logging
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import torch


default_init = nn.initializers.xavier_uniform


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class ScoreActor(nn.Module):
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module

    def __call__(self, obs_enc, actions, time, train=False):
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug(
                "Broadcasting obs_enc from %s to %s", obs_enc.shape, new_shape
            )
            obs_enc = jnp.broadcast_to(obs_enc, new_shape)

        reverse_input = jnp.concatenate([cond_enc, obs_enc, actions], axis=-1)

        # print("Cond Enc : ", cond_enc) # (5,32)
        # print("Obs Enc : ", obs_enc) # (5, 768)
        # print("Actions : ", actions) # (5, 30)
        # print("Reverse Input : ", reverse_input) # (5, 830)

        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred
    
class DiscreteScoreActor(nn.Module):
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module
    hidden_dim: int

    def setup(self):
        self.vocab_embed = DiscreteActionEmbeddingLayer(32, 2049)
        # self.rotary_emb = Rotary(self.hidden_dim)
        self.rotary_emb = Rotary(32)
        
    def apply_rotary(self, x, cos, sin):
        cos = cos[:, :, 0, 0, :]  # now shape (batch, seq, 32)
        sin = sin[:, :, 0, 0, :]  # now shape (batch, seq, 32)

        x1, x2 = jnp.split(x, 2, axis=-1)

        cos1, cos2 = jnp.split(cos, 2, axis=-1)
        sin1, sin2 = jnp.split(sin, 2, axis=-1)

        # return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return jnp.concatenate([x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2], axis=-1)

    def __call__(self, obs_enc, actions, time, train=False):
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug(
                "Broadcasting obs_enc from %s to %s", obs_enc.shape, new_shape
            )
            obs_enc = jnp.squeeze(obs_enc, axis=1)
            obs_enc = jnp.broadcast_to(obs_enc, new_shape)

        action_emb = self.vocab_embed(actions.astype(jnp.int32))
        cos, sin = self.rotary_emb(action_emb, seq_dim=1)

        # print(action_emb)
        # print(cos)

        rotated_action_emb = self.apply_rotary(action_emb, cos, sin)

        reverse_input = jnp.concatenate([cond_enc, obs_enc, rotated_action_emb], axis=-1)

        # print("Cond Enc : ", cond_enc) # (5,32)
        # print("Obs Enc : ", obs_enc) # (5, 768)
        # print("Actions : ", actions) # (5, 30)
        # print("Reverse Input : ", reverse_input) # (5, 830)

        eps_pred = self.reverse_network(reverse_input, train=train)
        return eps_pred


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jax.Array):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike, train: bool = False) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activation(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


class DiscreteActionEmbeddingLayer(nn.Module):
    dim: int = 32
    vocab_dim: int = 2049

    def setup(self):
        # Create a variance scaling initializer in 'fan_in' mode (similar to Kaiming uniform).
        initializer = jax.nn.initializers.variance_scaling(
            scale=2.0, mode='fan_in', distribution='uniform'
        )
        self.embedding = self.param('embedding', initializer, (self.vocab_dim, self.dim))

    def __call__(self, x):
        # x is assumed to be an array of indices.
        return self.embedding[x]

class Rotary(nn.Module):
    dim: int
    base: float = 10000.0

    def setup(self):
        # Compute inverse frequency vector: 1 / (base^(i/dim)) for even indices.
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))

    def __call__(self, x, seq_dim: int = 1):
        # x: input tensor; we assume x.shape[seq_dim] is the sequence length.
        seq_len = x.shape[seq_dim]
        # Create a range of positions for the sequence.
        t = jnp.arange(seq_len, dtype=self.inv_freq.dtype)
        # Compute the outer product: each position multiplied by each inverse frequency.
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate frequencies to have both cosine and sine parts.
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        # Compute cosine and sine, then add extra dimensions.
        cos_emb = jnp.cos(emb)[None, :, None, None, :]  # shape: (1, seq_len, 1, 1, dim)
        sin_emb = jnp.sin(emb)[None, :, None, None, :]
        # Tile along the third axis to mimic .repeat(1,1,3,1,1) in PyTorch.
        cos_cached = jnp.tile(cos_emb, (1, 1, 3, 1, 1))
        sin_cached = jnp.tile(sin_emb, (1, 1, 3, 1, 1))
        # For the third "head" (index 2), force the cosine values to 1 and sine values to 0.
        cos_cached = cos_cached.at[:, :, 2, :, :].set(1.0)
        sin_cached = sin_cached.at[:, :, 2, :, :].set(0.0)
        return cos_cached, sin_cached


def create_discrete_diffusion_model(
        out_dim : int,
        time_dim : int,
        num_blocks : int,
        dropout_rate : float,
        hidden_dim : int,
        use_layer_norm : bool,
):
    return DiscreteScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP((2 * time_dim, time_dim)),
        MLPResNet(
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
        hidden_dim
    )

def create_diffusion_model(
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP((2 * time_dim, time_dim)),
        MLPResNet(
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )