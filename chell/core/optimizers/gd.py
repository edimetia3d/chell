from typing import Iterable, Dict

import numpy as np

from chell.core import optimizer
from chell.core import tensor


class GD(optimizer.Optimizer):
    def __init__(self, params: Iterable[tensor.Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params)
        self.lr: float = lr
        self.momentum: float = momentum
        if momentum > 0:
            if momentum > 1:
                raise ValueError("momentum must be between 0 and 1")
            self.velocities = {}
            for p in params:
                self.velocities[p.node_name]: Dict[str, np.ndarray] = np.zeros(shape=p.value.shape)

    def step(self) -> None:
        for node_name, p in self.params.items():
            mean_grad = p.grad.mean(axis=0, keepdims=True)[0]
            if self.momentum > 0:
                self.velocities[node_name] = self.momentum * self.velocities[node_name] + self.lr * mean_grad
                p.value -= self.velocities[node_name]
            else:
                p.value -= self.lr * mean_grad
