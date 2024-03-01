from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from modules import QKVModule, attention


class SelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    w_o_init: Callable = nn.initializers.he_normal()

    @nn.compact
    def __call__(self, x):
        seq_length, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        z = jnp.zeros((seq_length, embed_dim * self.num_heads))
        for head in range(self.num_heads):
            q, v, k = QKVModule(in_features=embed_dim, out_features=embed_dim)(x)
            z.at[:, embed_dim * head:embed_dim * head + embed_dim].set(attention(q, k, v))

        return z @ self.param("w_o", self.w_o_init, (embed_dim * self.num_heads, embed_dim))
