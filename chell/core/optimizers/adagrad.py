from typing import Iterable, Dict

import numpy as np

from chell.core import optimizer
from chell.core import tensor


class AdaGrad(optimizer.Optimizer):
    def __init__(self, params: Iterable[tensor.Tensor], lr: float = 0.01):
        super().__init__(params)
        self.lr: float = lr
        self.grad_accumulator = {}
        for p in params:
            self.grad_accumulator[p.node_name]: Dict[str, np.ndarray] = np.zeros(shape=p.value.shape)

    def step(self) -> None:
        for node_name, p in self.params.items():
            p_grad = p.grad
            self.grad_accumulator[node_name] = self._accu_alpha() * self.grad_accumulator[
                node_name] + self._accu_beta() * p_grad * p_grad
            apply = self.lr / (np.sqrt(self.grad_accumulator[node_name]) + 1e-8) * p_grad
            p.value -= apply
            p._invalidate_user_value()

    def _accu_alpha(self):
        return 1

    def _accu_beta(self):
        return 1
