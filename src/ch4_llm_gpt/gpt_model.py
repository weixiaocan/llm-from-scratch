import torch
import torch.nn as nn
import tiktoken
from src.ch4_llm_gpt.transformer_ import TransformerBlock
from src.ch4_llm_gpt.layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 解包操作
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 预测单词
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


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
    model = GPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print(logits.shape)  # 输出形状应为 (batch_size, seq_len, vocab_size)
    # print(logits)

    # total_params = sum(p.numel() for p in model.parameters())
    # #模型的总参数数量
    # print(f"Total number of parameters: {total_params:,}")
    # # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    # total_size_bytes = total_params * 4

    # # Convert to megabytes
    # total_size_mb = total_size_bytes / (1024 * 1024)

    # print(f"Total size of the model: {total_size_mb:.2f} MB")
    # #计算总的容量

    # 模拟文本生成
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print(encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6,context_size=GPT_CONFIG_124M["context_length"])
    print("Output:",out)

    print("Output_length",len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)