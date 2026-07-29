import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
        fixed_data_boundaries=False,
        lazy_data_permutation=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True
        else:
            raise TypeError(
                "data_src must be a path, numpy array, or numpy memmap; "
                f"got {type(data_src).__name__}."
            )

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.fixed_data_boundaries = fixed_data_boundaries
        self.lazy_data_permutation = lazy_data_permutation

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker "
                f"{self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        self.last_epoch = None
        self.order = None
        self.permutation_offset = None
        self.permutation_stride = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        self.num_sequences = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Extra -1 to have a valid next token for the final start index.
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        # Construct the memmap each time to avoid a memory leak per NanoGPT.
        return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(np.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()
        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence so an offset remains valid when boundaries are not fixed.
        num_sequences = (len(self)) // self.sequence_length - 1
        if num_sequences < self.batch_size:
            raise ValueError(
                "Dataset is too small for one complete batch: "
                f"{num_sequences} sequences available, batch_size={self.batch_size}."
            )
        self.num_sequences = num_sequences
        if self.lazy_data_permutation:
            # An affine permutation avoids allocating an O(dataset size) index array.
            self.order = None
            self.permutation_offset = int(rng.integers(num_sequences))
            stride = int(rng.integers(1, num_sequences))
            while math.gcd(stride, num_sequences) != 1:
                stride = (stride + 1) % num_sequences
                if stride == 0:
                    stride = 1
            self.permutation_stride = stride
        else:
            self.order = rng.permutation(num_sequences)
        self.epoch_offset = (
            0 if self.fixed_data_boundaries else rng.integers(self.sequence_length)
        )
        self.last_epoch = epoch
        self.num_batches_of_seqlen = num_sequences // self.batch_size

    def _sample_without_replacement(self, step):
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen
        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        if self.lazy_data_permutation:
            positions = np.arange(start, end, dtype=np.int64)
            sequence_ids = (
                self.permutation_offset + positions * self.permutation_stride
            ) % self.num_sequences
        else:
            sequence_ids = self.order[start:end]
        return sequence_ids * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen

    @property
    def unique_tokens_per_epoch(self):
        """Target tokens exposed before the no-replacement reader repeats."""
        if self.with_replacement:
            raise ValueError(
                "unique_tokens_per_epoch is undefined for sampling with replacement."
            )
        return self.num_batches_of_seqlen * self.batch_size * self.sequence_length
