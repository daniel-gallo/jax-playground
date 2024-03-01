import flax.linen as nn

from modules import TransformerBlock, Linear


class Transformer(nn.Module):
    num_blocks: int
    num_heads: int
    output_dim: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = Linear(features=self.embed_dim)(x)

        for block in range(self.num_blocks):
            x = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)(x)

        return Linear(features=self.output_dim)(nn.LayerNorm()(x))