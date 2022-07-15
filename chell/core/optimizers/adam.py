from typing import Iterable, Dict

import numpy as np

from chell.core import optimizer
from chell.core import tensor


class Adam(optimizer.Optimizer):
    def __init__(self, params: Iterable[tensor.Tensor], lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99):
        super().__init__(params)
        self.lr: float = lr
        self.v_accumulator = {}
        self.a_accumulator = {}
        for p in params:
            self.v_accumulator[p.node_name]: Dict[str, np.ndarray] = np.zeros(shape=p.value.shape)
            self.a_accumulator[p.node_name]: Dict[str, np.ndarray] = np.zeros(shape=p.value.shape)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self) -> None:
        for node_name, p in self.params.items():
            p_grad = p.grad
            self.v_accumulator[node_name] = self.beta1 * self.v_accumulator[
                node_name] + (1 - self.beta1) * p_grad
            self.a_accumulator[node_name] = self.beta2 * self.a_accumulator[node_name] + (
                    1 - self.beta2) * p_grad * p_grad
            apply = self.lr * self.v_accumulator[node_name] / (np.sqrt(self.a_accumulator[node_name]) + 1e-8)
            p.value -= apply
            p._invalidate_user_value()
