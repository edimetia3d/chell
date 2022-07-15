from typing import Dict

import numpy as np

from chell.core import op
from chell.core import tensor


class FullyConnect(op.Operation):

    def __init__(self, M_W: int, M_X: int, N_X: int, x: op.OpArgT, Act: op.OperationClassVar):
        w = tensor.Tensor("fc.weight", np.random.randn(M_W, M_X), requires_grad=True)
        b = tensor.Tensor("fc.bias", np.random.randn(M_W, N_X), requires_grad=True)
        delegate = Act(w @ x + b)
        super().__init__("fc", {"delegate": delegate})

    def _compute(self) -> np.ndarray:
        v = self.inputs["delegate"]
        return v.value

    def _jacobian(self) -> Dict[str, np.ndarray]:
        v = self.inputs["delegate"]
        return {"delegate": np.eye(v.value.size)}
