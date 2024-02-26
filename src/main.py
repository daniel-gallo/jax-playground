import jax
from flax import linen as nn
from jax import random, numpy as jnp

from modules import Linear
from modules.qkv import QKVModule


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = Linear(in_features=784, out_features=128)(x)
        x = nn.relu(x)
        x = Linear(in_features=128, out_features=10)(x)
        q, k, v = QKVModule(in_features=10, out_features=10)(x)


model = Model()

key, subkey = random.split(random.PRNGKey(seed=42))
dummy_input = jnp.ones((64, 784))
params = model.init(subkey, dummy_input)
print(jax.tree_map(lambda x: x.shape, params))
