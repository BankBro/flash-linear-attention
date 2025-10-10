from torch.utils.data import DistributedSampler, RandomSampler
import torch
import math

# 导入日志工具
from train.utils.logger_utils import logger


class CheckpointingRandomSampler(RandomSampler):
    """
    pytorch lightning 每个epoch 都会调用一次train_dataloader(), 进而重新初始化 sampler,
    不管是正常训练流程还是断点续训, seed的采样顺序都一致, 保证可复现性。
    """
    def __init__(self, *args, **kwargs):
        logger.info(f"init sampler: {self.__class__.__name__}")

        super().__init__(*args, **kwargs)

        self.total_indices_len = len(self.data_source)

        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator().manual_seed(seed)

        self.random_state = self.generator.get_state()
        self.counter_index = 0

    def state_dict(self):
        logger.info(f"state dict: {self.__class__.__name__}")
        
        return {
            'random_state': self.random_state,
            'counter_index': self.counter_index,
        }

    def load_state_dict(self, state_dict):
        logger.info(f"load state dict: {self.__class__.__name__}")

        self.random_state = state_dict['random_state']
        self.counter_index = state_dict['counter_index']

        self.generator.set_state(self.random_state)

    def __iter__(self):
        indices = torch.randperm(self.total_indices_len, generator=self.generator).tolist()
        indices = indices[self.counter_index:]

        for index in indices:
            self.counter_index += 1
            assert self.counter_index <= self.total_indices_len, \
                f"Counter index {self.counter_index} exceeds total indices length {self.total_indices_len}."
            yield index

        self.counter_index = 0


class CheckpointingDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter_index = 0
    
    def state_dict(self):
        logger.info(f"state dict: {self.__class__.__name__}")
        return {
            'epoch': self.epoch,
            'counter_index': self.counter_index,
        }

    def load_state_dict(self, state_dict):
        logger.info(f"load state dict: {self.__class__.__name__}")
        self.epoch = state_dict['epoch']
        self.counter_index = state_dict['counter_index']

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        indices = indices[self.counter_index:]
        for index in indices:
            self.counter_index += 1
            assert self.counter_index <= self.total_size, \
                f"Counter index {self.counter_index} exceeds total size {self.total_size}."
            yield index
        
        self.counter_index = 0