import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True

        #最大的position应该是96，hidden_size就是embedding之后的维度数量
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

        elif x.dim() == 4:
            # return x + self.embeddings(position_ids)[None, None, :, :]
            h = x.size(1)

            #再这个维度变换的过程中，实际上只是数字位置进行了变换，但是元素的数量并未发生改变，所以完全可以进行这样的操作
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x



class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :] # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2) # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos)) # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


class RelativePositionEmbedding(nn.Module):
    def __init__(self, 
                 relative_attention_num_buckets, num_attention_heads, 
                 hidden_size, position_embedding_type):

        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.position_embedding_type = position_embedding_type
        self.num_attention_heads = num_attention_heads
        self.is_absolute = False

        if position_embedding_type == 'bias':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, num_attention_heads)

        elif position_embedding_type == 'contextual(1)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)
            self.to_r = nn.Linear(hidden_size, hidden_size, bias=False)

        elif position_embedding_type == 'contextual(2)':
            self.embeddings = nn.Embedding(relative_attention_num_buckets, hidden_size)

    def compute_bias(self, q, k, to_q=None, to_k=None):
        """
        q, k: [b h l d]
        return [b h l l]
        """
        h = self.num_attention_heads
        query_position = torch.arange(q.size(2), dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position   = torch.arange(k.size(2), dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets
        )

        if self.position_embedding_type == 'bias':
            bias = self.embeddings(relative_position_bucket)
            bias = rearrange(bias, 'm n h -> 1 h m n')

        elif self.position_embedding_type == 'contextual(1)':
            r = self.embeddings(relative_position_bucket)
            r = self.to_r(r)
            r = rearrange(r, 'm n (h d) -> h m n d', h=h)

            bias = torch.einsum('b h m d, h m n d -> b h m n', q, r)

        elif self.position_embedding_type == 'contextual(2)':
            r = self.embeddings(relative_position_bucket)

            kr = to_k(r)
            qr = to_q(r)

            kr = rearrange(kr, 'm n (h d) -> h m n d', h=h)
            qr = rearrange(qr, 'm n (h d) -> h m n d', h=h)

            bias1 = torch.einsum('b h m d, h m n d -> b h m n', q, kr)
            bias2 = torch.einsum('b h n d, h m n d -> b h m n', k, qr)

            bias = bias1 + bias2

        return bias

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance=128):
        """
        relative_position: [m n]
        """

        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
