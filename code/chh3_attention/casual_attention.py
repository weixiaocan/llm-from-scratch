import torch
import torch.nn as nn



# 用线性层简化实现

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        # 初始化因果自注意力模块
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 创建一个下三角矩阵作为掩蔽矩阵
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
    
        # 注意力分数是查询和键的点积
        attn_scores = queries @ keys.transpose(1,2)
        # 使用掩蔽矩阵将对角线以上的注意力分数设置为负无穷，可以根据input的长度进行调整
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
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

    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)

    attn_scores = torch.matmul(queries, keys.T)
    d_k = keys.shape[1]

    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
    # 在fostmax之后用一个下三角矩阵将元素之后的权重变为0，然后再次归一化
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    mask_simple = attn_weights * mask_simple
    rows_sumx = mask_simple.sum(dim=-1, keepdim=True)
    mask_simple_norm = mask_simple / rows_sumx
    print(mask_simple_norm)

    # 在softmax之前将对角线以上的未归一化的注意力用负无穷进行掩蔽
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    attn_weights = torch.softmax(masked / d_k**0.5, dim=-1)
    print(attn_weights)

    # 使用Dropout防止过拟合
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6,6)
    print("Dropout example:\n", dropout(example))

    torch.manual_seed(123)
    print(dropout(attn_weights))

    # 实现一个简洁的因果自注意力
    batch = torch.stack((inputs, inputs), dim=0)

    print("Batch inputs:\n", batch)

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)

    context_vec = ca(batch)
    print(context_vec)
    print("Context vector shape:", context_vec.shape)