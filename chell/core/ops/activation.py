from typing import Union, Dict, List

import numpy as np

from chell.core import op


class _Activation(op.Operation):
    _OP_NAME: str = None

    def __init__(self, x: op.OpArgT, ):
        op_name = self._OP_NAME
        if op_name is None:
            op_name = self.__class__.__name__.lower()
        super().__init__(op_name, {"x": x})



class Logistic(_Activation):

    def _compute(self) -> np.ndarray:
        ix = self.inputs["x"]
        return 1 / (1 + np.exp(-ix.value))

    def _jacobian(self) -> Dict[str, np.ndarray]:
        act = self.value
        jac = {}
        jac["x"] = act * (1 - act)
        return jac


class SoftMax(_Activation):

    def __init__(self, x: op.OpArgT, softmax_axis=None):
        self.softmax_axis = softmax_axis
        super().__init__(x)

    def _compute(self) -> np.ndarray:
        ix = self.inputs["x"]
        a = ix.value.copy()
        a[a > 1e2] = 1e2
        exp_v = np.exp(a)
        return exp_v / (np.sum(exp_v, axis=self.softmax_axis, keepdims=True) + 1e-10)

    def _jacobian(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError  # TODO, add it when needed


class LeakyRelu(_Activation):
    __LEAKY_RELU_ALPHA = 1e-4

    def _compute(self) -> np.ndarray:
        ix = self.inputs["x"].value.copy()
        ix[ix < 0] = ix[ix < 0] * LeakyRelu.__LEAKY_RELU_ALPHA
        return ix

    def _jacobian(self) -> Dict[str, np.ndarray]:
        act = self.value.copy()
        act[act > 0] = 1
        act[act < 0] = LeakyRelu.__LEAKY_RELU_ALPHA
        return {"x": np.diag(act.ravel())}
