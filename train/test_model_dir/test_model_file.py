from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn as nn
import torch

class SimpleGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        max_seq_length,
    ):
        super().__init__()
        
        # 使用 GPT2 的预定义配置和模型
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_head=nhead,
            n_layer=num_layers,
            n_positions=2048
        )
        self.transformer = GPT2LMHeadModel(self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        return outputs.logits


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 测试模型初始化
    model = SimpleGPT(
        vocab_size=50257,  # GPT-2 的词汇表大小
        d_model=768,       # 模型的隐藏层维度
        nhead=12,          # 注意力头数
        num_layers=12,     # Transformer 层数
        max_seq_length=1024  # 最大序列长度
    )
    model.to(device)

    # 打印模型结构
    print(model)

    # 测试前向传播
    input_ids = torch.randint(0, 50257 - 1, (8, 512)).to(device)  # 随机生成一个输入序列
    logits = model(input_ids)
    print(logits.shape)  # 应该是 (8, 512, 50257)

    # 测试generate方法 - 现在直接使用transformer的generate方法
    generated_ids = model.transformer.generate(
        input_ids,
        max_length=2048,
        min_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    print("Generated ids shape:", generated_ids.shape)  # 应该是 (1, <=2048)
    print("Generated ids:", generated_ids)
