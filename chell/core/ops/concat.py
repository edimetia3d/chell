from typing import Dict, Iterable

import numpy as np

from chell.core import op


class Concat(op.Operation):

    def __init__(self, xs: Iterable[op.OpArgT], axis=0):
        x_values = {}
        for i, x in enumerate(xs):
            x_values[f"x{i}"] = x
        self.axis = axis
        super().__init__("concat", x_values)

    def _compute(self) -> np.ndarray:
        xs_np = []
        for i in range(len(self.inputs)):
            x = self.inputs[f"x{i}"].value
            xs_np.append(x)
        return np.concatenate(xs_np, axis=self.axis)

    def _jacobian(self) -> Dict[str, np.ndarray]:
        accu_ind = 0
        final_shape = self.value.shape
        jac = {}
        axis = self.axis
        for i in range(len(self.inputs)):
            input_i = self.inputs[f"x{i}"].value
            if i != 0:
                accu_ind += self.inputs[f"x{i - 1}"].value.shape[self.axis]
            jac_i = np.zeros((self.value.size, input_i.size))
            indexes = list(np.unravel_index(np.arange(input_i.size), input_i.shape))
            indexes[self.axis] += accu_ind
            rows = np.ravel_multi_index(indexes, final_shape)
            for ind in zip(rows, range(input_i.size)):
                jac_i[ind[0], ind[1]] = 1
            jac[f"x{i}"] = jac_i
        return jac
