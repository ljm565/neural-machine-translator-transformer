from collections import Counter
from utils_func import load_dataset



class Tokenizer:
    def __init__(self, config, trainset_path, src:bool):
        self.data_id = 0 if src else 1
        self.vocab_size = config.vocab_size
        self.pad_token, self.sos_token, self.eos_token, self.unk_token = '[PAD]', '[SOS]', '[EOS]', '[UNK]'
        self.pad_token_id, self.sos_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self.word2idx = {self.pad_token: self.pad_token_id, self.sos_token: self.sos_token_id, self.eos_token: self.eos_token_id, self.unk_token: self.unk_token_id}
        self.idx2word = {self.pad_token_id: self.pad_token, self.sos_token_id: self.sos_token, self.eos_token_id: self.eos_token, self.unk_token_id: self.unk_token}

        # count the word frequency
        self.word_freq = Counter()
        for s in [d[self.data_id].split() for d in load_dataset(trainset_path)]:
            self.word_freq.update(s)

        # update vocab
        for word, _ in self.word_freq.most_common(self.vocab_size-len(self.word2idx)):
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

        assert len(self.word2idx) == len(self.idx2word)
        self.vocab_size = min(self.vocab_size, len(self.word2idx))


    def tokenize(self, s):
        return s.split()


    def encode(self, s):
        s = [self.word2idx[w] if w in self.word2idx else self.word2idx[self.unk_token] for w in self.tokenize(s)]
        return s


    def decode(self, tok):
        s = [self.idx2word[t] for t in tok]
        try:
            s = ' '.join(s[:tok.index(self.eos_token_id)])
        except ValueError:
            try:
                s = ' '.join(s[:tok.index(self.pad_token_id)])
            except ValueError:
                s = ' '.join(s)
        return s