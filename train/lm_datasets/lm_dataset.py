import torch
import numpy as np
import math


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, sample_len):
        assert isinstance(tokens, np.memmap), "tokens must be a memmap"
        self.tokens = tokens
        self.sample_len = sample_len

        """
        去掉最后那些不够一个完整sample_len的token, 只保留能被完整划分的token数量,
        但要额外保留一个token用于构建目标输出(target)
        """
        num_tokens = len(tokens)
        num_tokens = ((num_tokens - 1) // self.sample_len) * self.sample_len + 1
        self.num_tokens = num_tokens

        self.num_sample = math.ceil((self.num_tokens - 1) / self.sample_len)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        start_idx = idx * self.sample_len
        data = torch.as_tensor(
            self.tokens[start_idx:(start_idx + self.sample_len + 1)].astype(np.int64)
        )
        return data[:-1], data[1:].clone()