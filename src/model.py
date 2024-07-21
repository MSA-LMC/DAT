import torch
import torch.nn as nn
import torch.nn.functional as F

from src.position_embedding import *

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob,pos_emb):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.pos_emb = pos_emb

        
    def forward(self, query, key, value,mask, to_q, to_k):
        """
        query, key, value: [b h l d]
        mask: [b l]
        """
        batch_size = query.shape[0]
        if  self.pos_emb is not None and self.pos_emb.is_absolute is True:
            query = self.pos_emb(query)
            key = self.pos_emb(key)

        dots = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).float())

        if self.pos_emb is not None and self.pos_emb.is_absolute is False:
            bias = self.pos_emb.compute_bias(query, key, to_q, to_k)
            dots = dots + bias

        # Apply softmax
        attention = F.softmax(dots, dim=-1)
        attention = self.dropout(attention)
        # Calculate output
        out = torch.matmul(attention, value)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        return out
    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,pos_emb=None, dropout_prob=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.concat_heads = nn.Linear(embed_dim, embed_dim)

        self.attn_fn = Attention(embed_dim, num_heads,dropout_prob,pos_emb)
        self.dropout = nn.Dropout(dropout_prob)

        self.pos_emb = pos_emb

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]

        # Linear transformation for query, key, and value
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)


        # Reshape query, key, and value
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        out = self.attn_fn(query, key, value, mask, to_q=self.query, to_k=self.key)
        out = self.concat_heads(out)
        out = self.dropout(out)

        return out


class CEAMBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, pos_emb, rate=0.1):
        super(CEAMBlock, self).__init__()
        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

        self.pos_emb = pos_emb

    def forward(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class CEAMBlockwithCross(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, pos_emb, rate=0.1):
        super(CEAMBlockwithCross, self).__init__()
        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

        self.pos_emb = pos_emb

    def forward(self, inputs, partnet_inputs):
        attn_output = self.att(partnet_inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class Embedddings(nn.Module):
    def __init__(self, embed_dim,position_embedding_type,max_position_embeddings):
        super().__init__()
        self.dense = nn.Linear(2477, embed_dim)
        
        if position_embedding_type == 'learnable':
            self.position_embeddings = LearnableAbsolutePositionEmbedding(
                max_position_embeddings=max_position_embeddings, 
                hidden_size=embed_dim
            )
        
        elif position_embedding_type in ('fixed', 'rope'):
            self.position_embeddings = FixedAbsolutePositionEmbedding(
                max_position_embeddings=max_position_embeddings,
                hidden_size=embed_dim,
                position_embedding_type=position_embedding_type
            )

    def forward(self,embeds):
        embeds = self.dense(embeds)
        if hasattr(self, 'position_embeddings'):
            embeds = self.position_embeddings(embeds)

        return embeds
    
class CrossenhancedCEAM(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, N, M, K, dropout_rate,position_embedding_type,max_position_embeddings, alpha):
        super(CrossenhancedCEAM, self).__init__()
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.N = N
        self.M = M
        self.K = K
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings

        self.embedding = Embedddings(embed_dim,self.position_embedding_type,self.max_position_embeddings)
        self.embedding2 = Embedddings(embed_dim,self.position_embedding_type,self.max_position_embeddings)
        dim_heads = self.hidden_size // self.num_heads
        if self.position_embedding_type == 'layerwise_learnable':
            self.position_embeddings = LearnableAbsolutePositionEmbedding(
                max_position_embeddings=self.max_position_embeddings, 
                # hidden_size=dim_heads
                hidden_size=self.hidden_size
            )
        
        elif self.position_embedding_type in ('layerwise_fixed', 'layerwise_rope'):
            self.position_embeddings = FixedAbsolutePositionEmbedding(
                max_position_embeddings=self.max_position_embeddings,
                hidden_size=dim_heads,
                position_embedding_type=self.position_embedding_type.split('_')[-1],
            )

        elif self.position_embedding_type in ('layerwise_bias', 'layerwise_contextual(1)', 'layerwise_contextual(2)'):
            relative_attention_num_buckets = self.max_position_embeddings*2
            self.position_embeddings = RelativePositionEmbedding( 
                 relative_attention_num_buckets , 
                 self.num_heads, 
                 self.hidden_size, 
                 position_embedding_type=self.position_embedding_type.split('_')[-1],
                 # to_q=self.to_q,
                 # to_k=self.to_k
            )

        else:
            self.position_embeddings = None

        self.SA_blocks = nn.ModuleList([
            CEAMBlock(embed_dim, num_heads, ff_dim, self.position_embeddings, dropout_rate) for _ in range(self.N)
        ])

        self.SA_blocks2 = nn.ModuleList([
            CEAMBlock(embed_dim, num_heads, ff_dim, self.position_embeddings, dropout_rate) for _ in range(self.M)
        ])

        self.Cross_blocks = nn.ModuleList([
            CEAMBlockwithCross(embed_dim, num_heads, ff_dim, self.position_embeddings, dropout_rate) for _ in range(self.K)
        ])


        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.SELU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs, partnet_inputs):
        x = self.embedding(inputs)
        x = F.layer_norm(x, x.size()[1:])

        x2 = self.embedding2(partnet_inputs)
        x2 = F.layer_norm(x2, x2.size()[1:])

        for k in range(self.N):
            x_old = x
            x = self.SA_blocks[k](x)
            x = self.alpha * x + (1-self.alpha) * x_old  # Skip connection

        for k in range(self.M):
            x_old = x2
            x2 = self.SA_blocks2[k](x2)
            x2 = self.alpha * x2 + (1-self.alpha) * x_old  # Skip connection

        for k in range(self.K):
            x_old = x
            x = self.Cross_blocks[k](x,x2)
            x = self.alpha * x + (1-self.alpha) * x_old  # Skip connection
        
        x = self.regression_head(x)
        x = x.squeeze(-1)
        return x

'''model = CrossenhancedCEAM(embed_dim=768,
                          num_heads=8,
                          ff_dim=1024,
                          N=1,
                          M=1,
                          K=2,
                          dropout_rate=0.25,
                          position_embedding_type='layerwise_learnable',
                          max_position_embeddings=768,
                          alpha=0.5)
x1=torch.rand(8,96,2477)
x2=torch.rand(8,96,2477)
y=model(x1,x2)
print(y.size())'''