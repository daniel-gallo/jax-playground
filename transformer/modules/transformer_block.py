import flax.linen as nn

from modules import Linear
from modules.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        self_attention = SelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        x_prime = nn.LayerNorm()(x)
        x_prime_prime = x + self_attention(x_prime)

        x_prime_prime_prime = nn.LayerNorm()(x_prime_prime)
        return x_prime_prime + nn.Sequential([
            Linear(features=self.embed_dim),
            nn.gelu,
            Linear(features=self.embed_dim)
        ])(x_prime_prime_prime)
