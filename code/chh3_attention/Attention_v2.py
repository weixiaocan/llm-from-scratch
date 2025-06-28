import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
#对于一句话中的每个单词定义了一个三维的向量

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out),requires_grad=False)

query_2 = x_2 @ W_query  # 计算查询向量
keys = inputs @ W_key
values = inputs @ W_value

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)  # 计算注意力分数
print("Attention score for x^2:", attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("Attention scores for x^2:\n", attn_scores_2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) #利用维度进行点积的缩放
print("Attention weights for x^2:\n", attn_weights_2)


# 计算每一个token向量
context_vec_2 = attn_weights_2 @ values
print("Context vector for x^2:\n", context_vec_2)

# 自注意力模块

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in,d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_vl = SelfAttention_v1(d_in, d_out)
print(sa_vl(inputs))

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
    
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
