import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader



class GPRDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # return self.input_ids[idx], self.target_ids[idx]
        return (
        torch.tensor(self.input_ids[idx], dtype=torch.long),
        torch.tensor(self.target_ids[idx], dtype=torch.long)
    )
    

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPRDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers)

    return dataloader

with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text = f.read()


vocab_size = 50257
output_dim = 256

torch.manual_seed(123)  # For reproducibility
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)



token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)  # Should be (batch_size, max_length, output_dim)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embedding = token_embeddings + pos_embeddings
print(input_embedding.shape)  # Should be (batch_size, max_length, output_dim)