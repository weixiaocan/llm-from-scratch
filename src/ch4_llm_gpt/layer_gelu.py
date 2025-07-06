import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GELU(nn.Module):
    # GELU激活函数的实现
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor( 2.0 / torch.pi)) * 
            (x + 0.0444715 * torch.pow(x, 3))))
    




class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    # 运行一次就线性两次，激活一次
    def forward(self, x):
        return self.layers(x)
    

if __name__ == "__main__":

    GPT_CONFIG_124M = {"vocab_size":50357,
                   "context_length":1024,
                   "emb_dim" : 768,
                   "n_heads" : 12,
                   "n_layers" : 12,
                   "drop_rate" : 0.1,
                   "qkv_bias" : False
                   }
    
    # # 可视化GELU和ReLU激活函数
    # gelu, relu = GELU(), nn.ReLU()

    # x = torch.linspace(-3, 3, 100)
    # y_gelu, y_selu = gelu(x), relu(x)

    # plt.figure(figsize=(10, 5))
    # for i, (y, label) in enumerate(zip([y_gelu, y_selu], ["GELU", "ReLU"]), 1):
    #     plt.subplot(1, 2, i)
    #     plt.plot(x,y)
    #     plt.title(label)
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.grid()
        
    # plt.tight_layout()
    # plt.show()

    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2,3,768)
    out = ffn(x)
    print("out:", out.shape)