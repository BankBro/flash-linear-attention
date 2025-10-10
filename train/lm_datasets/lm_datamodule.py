from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from transformers import AutoTokenizer
import time
from pathlib import Path
import numpy as np
from itertools import chain
import os
import mmap
import shutil
import pickle

# 导入日志工具
from train.utils.logger_utils import logger

from train.lm_datasets.lm_dataset import LMDataset
from train.lm_datasets.checkpointing_sampler import (
    CheckpointingDistributedSampler,
    CheckpointingRandomSampler
)
from omegaconf import DictConfig

MAX_UINT16_VOCAB_SIZE = 64 * 1024
TOKENIZER_MAX_LENGTH = int(1e30)


class LMDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_name: str,
        dataset_config_name: str,  # 数据集的具体配置版本
        tokenizer_name: str,
        cache_dir: str,
        max_length: int,
        add_eos: bool,
        batch_size_per_gpu: int,
        batch_size_eval: int,
        num_workers: int,          # DataLoader 的 worker 数量
        shuffle: bool,
        global_seed: int

    ):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else cache_dir
        self.max_length = max_length
        self.add_eos = add_eos
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.global_seed = global_seed
        self.ddp = True if len(self.cfg['trainer']['devices']) > 1 else False

        self.tokenizer_file_name = 'tokenizer.pkl'
        self.tokenized_cache_dir = Path(self.cache_dir) / self.tokenized_dataset_name

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        # prepare_data()后加载tokenizer可获取vocab_size
        self.prepare_data()
        self.tokenizer = self._load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # prepare_data()后可获取全部ids(tokens)
        concat_ids = self._load_concat_ids_as_mmap()
        if self.cfg['debug']['flag']:
            self.ids_num_train = self.cfg['debug']['ids_num_train']
            self.ids_num_val = self.cfg['debug']['ids_num_val']
            self.ids_num_test = self.cfg['debug']['ids_num_test']
        else:
            self.ids_num_train = len(concat_ids['train'])
            self.ids_num_val = len(concat_ids['validation'])
            self.ids_num_test = len(concat_ids['test'])

    @property
    def tokenized_dataset_name(self):
        return (
            f'tokenizer-{self.tokenizer_name}'
            f'--dataset-{self.dataset_name}'
            f'--dataset_config-{self.dataset_config_name}'
            f'--max_length-{self.max_length}'
            f'--add_eos-{self.add_eos}'
        )
    

    def load_state_dict(self, checkpoint):
        """
        断点续训时pytorch lightning调用顺序:
        __init__ -> load_state_dict -> prepare_data -> setup -> train_dataloader

        正常训练时pytorch lightning调用顺序:
        __init__ -> prepare_data -> setup -> train_dataloader
        """
        
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']

    def _print_dataset_info(self, raw_dataset):
        logger.info(f'Dataset({self.dataset_name}), '
              f'config dataset({self.dataset_config_name}) has been loaded.')
        
        for name, sub_dataset in raw_dataset.items():
            logger.info(f'{name}:\t{sub_dataset.num_rows} rows.')

    def _load_dataset_with_cache_dir(self):
        logger.info(f"loading dataset: {self.dataset_name}...")
        
        time_begin = time.time()
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir
        )
        time_end = time.time()

        logger.info(f"load dataset time: {(time_end - time_begin):.3f}s")
        return dataset
    
    def _tokenize_dataset(self, raw_dataset, tokenizer, dtype):
        def tokenize_concat_samples(examples):
            if self.add_eos:
                add_eos_seq = lambda seq:(seq + tokenizer.eos_token) if seq else seq
                add_eos_seqs = lambda seqs: [add_eos_seq(seq) for seq in seqs]
                tokenize = lambda example: tokenizer(add_eos_seqs(example["text"]))
            else:
                tokenize = lambda example: tokenizer(example["text"])
            
            input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)

            return {'input_ids': [input_ids], 'len': [len(input_ids)]}
            
        tokenized_dataset = raw_dataset.map(
            tokenize_concat_samples,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=["text"],
            desc="Running tokenizer on dataset",
        )

        return tokenized_dataset
    
    def _concat_and_save_ids(self, tokenized_dataset, dtype):
        concat_ids = {}
        
        for name, dataset in  tokenized_dataset.items():
            cum_len_arr = np.cumsum(dataset['len'])
            array_len = cum_len_arr[-1]
            tokenized_dataset[name] = dataset.add_column('cum_len', cum_len_arr)

            file_name = f'{name}.bin'
            tmp_file_path = self.tokenized_cache_dir / file_name

            with open(tmp_file_path, 'wb') as f:
                os.posix_fallocate(f.fileno(), 0, array_len * np.dtype(dtype).itemsize)
                # logger.info(f'Created file {tmp_file_path}')
            
            def write_to_file(sample):
                with open(tmp_file_path, 'rb+') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    start_idx = sample['cum_len'] - sample['len']
                    sample_len = sample['len']

                    arr = np.ndarray((sample_len,), dtype=dtype, buffer=mm,
                                     offset=np.dtype(dtype).itemsize * start_idx)
                    arr[:] = sample['input_ids']
            
            # 将所有样本写入文件时拼接
            tokenized_dataset[name].map(
                write_to_file,
                batched=False,
                num_proc=self.num_workers,
                desc='Concatenating samples'
            )

            # 当所有样本写入文件后再落盘
            with open(tmp_file_path, 'r+') as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm.flush()
                os.fsync(f.fileno())

            concat_ids[name] = np.memmap(tmp_file_path, dtype=dtype, mode='r', shape=(array_len,))
            # logger.info(f'{file_name} has been written to disk.')
        
        return concat_ids

    def _change_concat_ids_as_npy(self, concat_ids):
        assert self.tokenized_cache_dir.exists(), 'tokenized_cache_dir does not exist.'

        for name, ids in concat_ids.items():
            np.save(self.tokenized_cache_dir / f'{name}.npy', ids)

            tmp_bin_file = self.tokenized_cache_dir / f'{name}.bin'
            if tmp_bin_file.exists():
                tmp_bin_file.unlink()
            
            # logger.info(f'Saved {name}.npy to disk.')
    
    def _save_tokenizer(self, tokenizer):
        with open(self.tokenized_cache_dir / self.tokenizer_file_name, 'wb') as f:
            pickle.dump(tokenizer, f)

    def prepare_data(self):
        assert self.cache_dir is not None, "cache_dir must be set"
        
        if self.tokenized_cache_dir.exists():
            logger.info(f"Preparing data has been done.")
            return
    
        try:
            self.tokenized_cache_dir.mkdir(parents=True, exist_ok=True)

            raw_dataset = self._load_dataset_with_cache_dir()
            self._print_dataset_info(raw_dataset)

            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
            tokenizer.model_max_length = TOKENIZER_MAX_LENGTH
            logger.info(f"Tokenizer({self.tokenizer_name}) has loaded.")

            dtype = np.uint16 if tokenizer.vocab_size < MAX_UINT16_VOCAB_SIZE else np.int32
            tokenized_dataset = self._tokenize_dataset(raw_dataset, tokenizer, dtype)

            concat_ids = self._concat_and_save_ids(tokenized_dataset, dtype)
            self._change_concat_ids_as_npy(concat_ids)
            logger.info("Saved concatenate tokenized dataset to disk as npy file.")

            self._save_tokenizer(tokenizer)
            logger.info('Saved tokenizer to disk.')
            logger.info('Processing dataset done.')

        except Exception as e:
            logger.info('Something went wrong while preparing the data.')

            if self.tokenized_cache_dir.exists():
                shutil.rmtree(self.tokenized_cache_dir)
            raise e

    def _load_concat_ids_as_mmap(self):
        concat_ids = {
            f'{name}': np.load(self.tokenized_cache_dir / f'{name}.npy', mmap_mode='r')
            for name in ['train', 'validation', 'test']
        }

        for name, ids in concat_ids.items():
            assert isinstance(ids, np.memmap), f'{name} is not a memmap'
            
        return concat_ids

    def _load_tokenizer(self):
        tokenizer_file_path = self.tokenized_cache_dir / self.tokenizer_file_name
        assert tokenizer_file_path.exists(), f'{self.tokenizer_file_name} not exist.'
        
        with open(tokenizer_file_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def setup(self, stage: str = None):
        assert self.tokenized_cache_dir.exists(), f'data has not been prepaered'

        concat_ids = self._load_concat_ids_as_mmap()

        if stage == 'fit' or stage is None:
            if self.dataset_train is None:
                self.dataset_train = LMDataset(
                    tokens = concat_ids['train'][:self.ids_num_train],
                    sample_len = self.max_length
                )

                self.dataset_val = LMDataset(
                    tokens = concat_ids['validation'][:self.ids_num_val],
                    sample_len = self.max_length
                )

                logger.info(f'Loaded training and validation datasets, stage: {stage}.')
        
        if stage == 'test' or stage is None:
            if self.dataset_test is None:
                self.dataset_test = LMDataset(
                    tokens = concat_ids['test'][:self.ids_num_test],
                    sample_len = self.max_length
                )
                logger.info(f'Loaded test dataset, stage: {stage}.')

        logger.info(f'Setup done, stage: {stage}.')

    def _data_loader(
            self,
            dataset = None,
            batch_size = None,
            shuffle = False,
            sampler = None,
            drop_last = True,
        ):
        assert dataset is not None and batch_size is not None

        return DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = 5,
            shuffle = shuffle,
            sampler = sampler,
            drop_last = drop_last,
            pin_memory = True
        )

    def train_dataloader(self):
        assert (
            (self.ddp and self.cfg['trainer']['strategy'] == 'ddp') or \
            (not self.ddp and self.cfg['trainer']['strategy'] != 'ddp')
        ), f"strategy={self.cfg['trainer']['strategy']}, ddp={self.ddp}"

        if self.ddp:
            sampler = CheckpointingDistributedSampler(
                self.dataset_train,
                seed = self.global_seed,
                drop_last = False
            )
            logger.info(f'Using CheckpointingDistributedSampler.')

            # 断点续训状态, pytorch lightning 会调用 load_state_dict
            if self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter_index': self.fast_forward_batches * self.batch_size_per_gpu
                })

        else:
            sampler = CheckpointingRandomSampler(self.dataset_train)
            logger.info(f'Using CheckpointingRandomSampler.')
        
        return self._data_loader(
            dataset = self.dataset_train,
            batch_size = self.batch_size_per_gpu,
            shuffle = False,
            sampler = sampler,
            drop_last = True
        )

    def val_dataloader(self):
        return self._data_loader(
            dataset = self.dataset_val,
            batch_size = self.batch_size_eval,
            drop_last = False
        )

    def test_dataloader(self):
        return self._data_loader(
            dataset = self.dataset_test,
            batch_size = self.batch_size_eval,
            drop_last = False
        )