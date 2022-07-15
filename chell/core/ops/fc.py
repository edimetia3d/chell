import numpy as np

from chell.core import op
from chell.core import tensor


def FullyConnect(M_W: int, M_X: int, N_X: int, x: op.Operation, Act: op.OperationClassVar) -> op.Operation:
    w = tensor.Tensor("fc.weight", np.random.randn(M_W, M_X), requires_grad=True)
    b = tensor.Tensor("fc.bias", np.random.randn(M_W, N_X), requires_grad=True)
    return Act(w @ x + b)
