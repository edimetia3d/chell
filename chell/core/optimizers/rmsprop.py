from typing import Iterable

from chell.core import tensor
from chell.core.optimizers import adagrad


class RMSProp(adagrad.AdaGrad):
    def __init__(self, params: Iterable[tensor.Tensor], lr: float = 0.01, beta: float = 0.9):
        super().__init__(params, lr)
        self.beta = beta

    def _accu_alpha(self):
        return self.beta

    def _accu_beta(self):
        return 1 - self.beta
