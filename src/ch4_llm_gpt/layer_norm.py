import torch
import torch.nn as nn





class LayerNorm(nn.Module):
    # 归一化函数，可以避免信息泄露也可以稳定
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  #避免除零错误
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和偏移
    

if __name__ == "__main__":

    batch_example = torch.randn(2,5)
    # 一个按顺序执行的神经网络
    layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("mean:", mean)
    print("var:", var)

    out_norm = (out - mean) / torch.sqrt(var)
    print("out_noorm:", out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    torch.set_printoptions(sci_mode=False)
    print("mean:", mean)
    print("var:", var)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased = False, keepdim=True)

    print("mean:", mean)
    print("var:", var)