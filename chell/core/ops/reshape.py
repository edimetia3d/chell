from typing import Dict

import numpy as np

from chell.core import op


class Reshape(op.Operation):

    def __init__(self, x: op.OpArgT, new_shape: op.OpArgT):
        super().__init__("reshape", {"x": x, "new_shape": new_shape})

    def _compute(self) -> np.ndarray:
        x = self.inputs["x"].value
        new_shape = self.inputs["new_shape"].value
        return x.reshape(tuple(new_shape))

    def _jacobian(self) -> Dict[str, np.ndarray]:
        x = self.inputs["x"].value
        return {"x": np.eye(x.size)}
