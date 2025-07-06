import torch
import torch.nn as nn
from src.ch3_attention.casual_attention import CausalAttention


# 由多个因果注意力机制叠加的实现方式
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 多个实例，每个实例是一个头
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])


    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# 实现独立的多头注意力机制

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # 初始化头的维度和数量
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #组合所有头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将查询、键和值分成多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 转换维度
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 注意力分数是查询和键的点积
        attn_scores = queries @ keys.transpose(2,3)
        # 使用掩蔽矩阵将对角线以上的注意力分数设置为负无穷
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        # 将多个头的输出拼接起来
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec



if __name__ == "__main__":

    inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

    batch = torch.stack((inputs, inputs), dim=0)
    # 实现一个简洁的多头自注意力
    torch.manual_seed(123)

    context_length = batch.shape[1]
    d_in, d_out = 3,2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    context_vecs = mha(batch)

    # 输出的形状应该是 (batch_size, context_length, num_heads * d_out)
    print("Context vectors shape:", context_vecs.shape)
    print("Context vectors:", context_vecs)

    # 测试独立实现的多头注意力机制
    torch.manual_seed(123)
    batch_size, context_length, d_om = batch.shape
    d_out = 2
    mha_v2 = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
    context_vecs_v2 = mha_v2(batch)
    print("Context vectors shape (v2):", context_vecs_v2.shape)
    print("Context vectors (v2):", context_vecs_v2)