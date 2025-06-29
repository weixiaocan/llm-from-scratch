import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
#对于一句话中的每个单词定义了一个三维的向量

query = inputs[1]

attn_scores = torch.empty(inputs.shape[0])

for i ,x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i,query)

print("Attention scores:", attn_scores)

def softmax_naive(x):
    return torch.exp(x) / torch.sum(torch.exp(x))

attn_weights_2_naive = softmax_naive(attn_scores)
attn_weights_2 = torch.softmax(attn_scores, dim=0)

context_vec_2 = torch.zeros(query.shape)
# 创造一个内容零向量
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# print("Context vector :", context_vec_2)


# 推广到所有输入上
attn_scores = torch.empty(6,6)

# 建立空表存储相关联程度
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# print("Attention scores matrix:\n", attn_scores)

# 用矩阵相乘实现
attn_scores = inputs @ inputs.T
print("Attention scores matrix:\n", attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention weights matrix:\n", attn_weights)

print(attn_weights.sum(dim=-1))  # 每行的和应该为1

all_conmtext_vecs = attn_weights @ inputs
print("All context vectors:\n", all_conmtext_vecs)