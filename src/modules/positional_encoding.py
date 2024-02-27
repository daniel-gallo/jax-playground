import jax.numpy as jnp
from flax import linen as nn


class PositionalEncoder(nn.Module):
    def __call__(self, sequence_length, embedding_dimension):
        assert embedding_dimension % 2 == 0

        position = jnp.arange(sequence_length)
        denominator = 10000 ** (jnp.arange(0, embedding_dimension, 2) / embedding_dimension)

        position = jnp.repeat(position[:, jnp.newaxis], repeats=embedding_dimension // 2, axis=1)
        denominator = jnp.repeat(denominator[jnp.newaxis, :], repeats=sequence_length, axis=0)

        positional_encoding = jnp.zeros((sequence_length, embedding_dimension))
        positional_encoding = positional_encoding.at[:, 0::2].set(jnp.sin(position / denominator))
        positional_encoding = positional_encoding.at[:, 1::2].set(jnp.cos(position / denominator))

        return positional_encoding
