import torch
import torch.nn as nn
from fla.models import GLAConfig
from transformers import AutoModelForCausalLM


class GLA(nn.Module):
    def __init__(self, *args, vocab_size = None, **kwargs):
        super().__init__()
        self.config = GLAConfig(*args, vocab_size = vocab_size, **kwargs)
        self.model = AutoModelForCausalLM.from_config(self.config)

    def forward(self, *args, **kwargs):
        output =  self.model(*args, **kwargs)
        return output.logits


if __name__ == '__main__':
    batch_size, seq_len = 4, 1024
    vocab_size = 32000
    device = 'cuda:0'

    model = GLA(vocab_size = vocab_size).to(device)
    print(model)

    x = torch.randint(100, vocab_size - 100, (batch_size, seq_len)).to(device=device)
    y = model(x)

    print(type(y))
    print(y.shape)
    # print(y.logits)