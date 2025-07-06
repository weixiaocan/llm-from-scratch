from src.ch3_attention.multi_head_attention import MultiHeadAttention
from torch import nn
import torch
from src.ch4_llm_gpt.layer_gelu import FeedForward
from src.ch4_llm_gpt.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads= cfg["n_heads"],
            dropout = cfg["drop_rate"],
            akv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shhortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 对前馈神经网络的残差连接
        shhortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shhortcut

        return x
    

if __name__ == "__main__":
    torch.manual_seed(123)
    GPT_CONFIG_124M = {"vocab_size":50357,
                       "context_length":1024,
                       "emb_dim" : 768,
                       "n_heads" : 12,
                       "n_layers" : 12,
                       "drop_rate" : 0.1,
                       "qkv_bias" : False
                       }
    
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    out = block(x)
    print(x.shape)
    print(out.shape)