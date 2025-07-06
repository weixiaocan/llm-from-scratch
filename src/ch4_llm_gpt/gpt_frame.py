import torch
import tiktoken
import torch.nn as nn

# 搭建一个gpt模型的框架，主要部分留空
class DummyGPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 使用多个transformer块
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 归一化层，调整输出分布1
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        # 输出层，映射到词汇表大小
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        # 词嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        # 词嵌入与位置嵌入相加
        x = tok_embeds + pos_embeds
        # 嵌入dropout
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # 输出层
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):

    def __init__(sself, cfg):
        super().__init__()
        # 占位，实际应该实现模型注意力机制和前馈网络


    def forward(self, x):
        # 占位，实际应该实现模型注意力机制和前馈网络
        return x
    

class DummyLayerNorm(nn.Module):

    def __init__(self, normalized_shhape, eps=1e-5):
        super().__init__()
        # 占位，实际应该实现层归一化

    
    def forward(self, x):
        # 占位，实际应该实现层归一化
        return x







if __name__ == "__main__":

    GPT_CONFIG_124M = {"vocab_size":50357,
                   "context_length":1024,
                   "emb_dim" : 768,
                   "n_heads" : 12,
                   "n_layers" : 12,
                   "drop_rate" : 0.1,
                   "qkv_bias" : False
                   }
    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []

    txt1 = "Every effort moves you"
    txt_2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt_2)))

    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits.shape)  # 输出形状应为 (batch_size, seq_len, vocab_size)
    print(logits)
                 