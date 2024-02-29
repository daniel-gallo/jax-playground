import jax.numpy as jnp
from flax import linen as nn


def attention(q, k, v):
    assert q.shape == k.shape == v.shape
    batch_size, embedding_dimension = q.shape

    # TODO: review
    return nn.softmax(
        q @ k.T / jnp.sqrt(embedding_dimension)
    ) @ v
