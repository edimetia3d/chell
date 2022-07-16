import numpy as np

from chell.core import op
from chell.core.ops import math


class Embedding(math.Matmul):
    """Embedding is a faster version of Matmul.

    Given that x is a one-hot colum vector, `W @ x` will be the W[i, :] * x[i]
    where i is the index of the non-zero element in the vector.

    Technically, any `W @ x` could be called embedding, embedding is a tool to convert a K dimensional vector to
    a M dimensional vector, where M is less than K, so the information is reduced in a lower dimension.

    e.g. When x is a 1,000,000,000 x 1 vector, it will be very hard to compute things related to it, but by using
    embedding, we may use a 100 * 1,000,000,000 `M` matrix to do a `M @ x`, then lower the dimension to 100, this will be much
    easier to handle.
    """

    def __init__(self, W: op.OpArgT, x: op.OpArgT):
        super().__init__(W, x)

    def _compute(self) -> np.ndarray:
        W = self.inputs["x"].value
        x = self.inputs["y"].value
        ind = np.argwhere(x != 0)
        assert len(ind) == 1, "Embedding only supports one-hot vectors"
        return W[[ind[0]], :] * x[ind[0]]
