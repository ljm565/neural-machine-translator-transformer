import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, tokenizers, config):
        self.data = data
        self.src_tokenizer, self.trg_tokenizer = tokenizers[0], tokenizers[1]
        self.max_len = config.max_len
        
        self.all_src, self.all_trg = [], []
        for src, trg in self.data:
            s = self.add_special_token(src, self.src_tokenizer)
            t = self.add_special_token(trg, self.trg_tokenizer)
            self.all_src.append(s)
            self.all_trg.append(t)

        assert len(self.all_src) == len(self.all_trg)
        self.length = len(self.all_src)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.sos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        return torch.LongTensor(self.all_src[idx]), torch.LongTensor(self.all_trg[idx])

    
    def __len__(self):
        return self.length