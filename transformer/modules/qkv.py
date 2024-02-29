from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class QKVModule(nn.Module):
    in_features: int
    out_features: int

    init_function: Callable = nn.initializers.he_normal()

    @nn.compact
    def __call__(self, x):
        # X is (bs, in_features)
        w_q = self.param(
            "w_q",
            self.init_function,
            (self.in_features, self.out_features)
        )
        w_k = self.param(
            "w_k",
            self.init_function,
            (self.in_features, self.out_features)
        )
        w_v = self.param(
            "w_v",
            self.init_function,
            (self.in_features, self.out_features)
        )

        q = jnp.dot(x, w_q)
        k = jnp.dot(x, w_k)
        v = jnp.dot(x, w_v)

        return q, k, v
