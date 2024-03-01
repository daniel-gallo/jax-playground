from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class Linear(nn.Module):
    features: int

    # The init functions should be like `lambda key, shape: jnp.random.normal(key, shape)`
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (in_features, self.features)
        )
        bias = self.param(
            "bias",
            nn.initializers.zeros_init(),
            (self.features,)
        )

        return jnp.dot(x, kernel) + bias
