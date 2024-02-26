from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class Linear(nn.Module):
    in_features: int
    out_features: int

    # The init functions should be like `lambda key, shape: jnp.random.normal(key, shape)`
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.in_features, self.out_features)
        )
        bias = self.param(
            "bias",
            nn.initializers.zeros_init(),
            (self.out_features,)
        )

        return jnp.dot(x, kernel) + bias
