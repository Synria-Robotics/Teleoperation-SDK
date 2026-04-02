from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: list[dict[str, np.ndarray | float | bool]] = []
        self.pos = 0

    def add(self, item: dict[str, np.ndarray | float | bool]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not self.storage:
            raise ValueError("ReplayBuffer is empty")
        idx = np.random.randint(0, len(self.storage), size=int(batch_size))
        batch = [self.storage[i] for i in idx]
        keys = batch[0].keys()
        out: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [sample[key] for sample in batch]
            arr = np.asarray(values, dtype=np.float32)
            out[key] = torch.as_tensor(arr, dtype=torch.float32)
        return out
