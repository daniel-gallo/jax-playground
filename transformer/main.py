import jax
from flax import linen as nn
from jax import random, numpy as jnp
import matplotlib.pyplot as plt

from modules import Linear, QKVModule, Attention, PositionalEncoder

pe = PositionalEncoder()(1000, 500)
plt.imshow(pe)
plt.show()
