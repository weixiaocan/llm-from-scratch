from importlib.metadata import version
import os
import re
import urllib.request


# print("torch version:", version("torch"))
# print("tiktoken version", version("tiktoken"))

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)##从指定的地点读取文件


with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;!?_"()\']|--|\s)', raw_text)
preprocessed = [s.strip() for s in preprocessed if s.strip()]
# print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])  # 添加未知和填充标记

vocab = {token:integer for integer, token in enumerate(all_words)}


class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_srt = {i:s for s,i in vocab.items()}

    def encode(self, text):
        processed_text = re.split(r'([,.:;!?_"()\']|--|\s)', text) ##正则化分词标点符号

        processed_text = [s.strip() for s in processed_text if s.strip()]
        processed_text = [s if s in self.str_to_int else "<|unk|>" for s in processed_text]
        ids = [self.str_to_int[s] for s in processed_text]
        return ids

    
    def decode(self,ids):
        text = " ".join([self.int_to_srt[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #使用正则表达式，去除标点符号前的多余空格
        # \s+匹配一个或者多个空白  \1 替换到匹配
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

text = "The verdict is not guilty, but the jury is still out on that."
ids = tokenizer.encode(text)
print("Encoded IDs:", ids)

tt = tokenizer.decode(ids)
print(tt)