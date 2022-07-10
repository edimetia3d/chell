# all operation in this file will be used in the op.Operation class

from typing import Callable, Any, Union, Optional

import numpy as np

# beg tensor must be imported first, so that the import chain  ops-> tensor -> op will be called in the correct order
from chell.core import tensor

del tensor
from chell.core import op


# end

class _BinaryOp(op.Operation):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = None
    _OP_NAME: str = None

    def __init__(self, x: op.OpArgT, y: op.OpArgT, ):
        op_name = self._OP_NAME
        if op_name is None:
            op_name = self.__class__.__name__.lower()
        super().__init__(op_name, {"x": x, "y": y})

    def _compute(self):
        ix = self.inputs["x"]
        iy = self.inputs["y"]
        return self.__class__._binary_np_func(ix.value, iy.value)


class _Reduce(op.Operation):
    _reduce_np_func: Callable[[Any, Optional[int]], np.ndarray] = None
    _OP_NAME: str = None

    def __init__(self, x: op.OpArgT, axis: Union[int, None]):
        op_name = self._OP_NAME
        if op_name is None:
            op_name = self.__class__.__name__.lower()
        self.axis = axis
        super().__init__(op_name, {"x": x})

    def _compute(self):
        ix = self.inputs["x"]
        v = self.__class__._reduce_np_func(ix.value, self.axis)
        if np.isscalar(v):
            v = np.array([v])
        return v


class ReduceSum(_Reduce):
    _reduce_np_func = np.sum

    def _jacobian(self):
        ix = self.inputs["x"]
        jac = {}
        jac["x"] = np.ones(shape=(1, ix.value.size))
        return jac


def _can_broadcast(v0: np.ndarray, v1: np.ndarray) -> bool:
    s0 = np.asarray(v0.shape)
    s1 = np.asarray(v1.shape)
    return ((s0 == 1) | (s1 == 1) | (s1 == s0)).all()


class Add(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.add

    def _jacobian(self):
        ix = self.inputs["x"]
        iy = self.inputs["y"]
        jac = {}
        assert _can_broadcast(ix.value, iy.value)
        jac["x"] = np.eye(self.value.size)
        jac["y"] = np.eye(self.value.size)
        return jac


class Sub(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.subtract

    def _jacobian(self):
        ix = self.inputs["x"]
        iy = self.inputs["y"]
        jac = {}
        assert _can_broadcast(ix.value, iy.value)
        jac["x"] = np.eye(self.value.size)
        jac["y"] = np.eye(self.value.size) * -1
        return jac


class Mul(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.multiply

    def _jacobian(self):
        ix = self.inputs["x"]
        iy = self.inputs["y"]
        jac = {}
        assert _can_broadcast(ix.value, iy.value)
        jac["x"] = np.diag(np.broadcast_to(iy.value, self.value.shape).ravel())
        jac["y"] = np.diag(np.broadcast_to(ix.value, self.value.shape).ravel())
        return jac


class Div(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.divide


class Mod(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.mod


class DivMod(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.divmod


class Eq(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.equal


class LE(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.less_equal


class GE(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.greater_equal


class LT(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.less


class GT(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.greater


class NE(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.not_equal


class AllClose(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.allclose


class And(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.logical_and


class Or(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.logical_or


class Pow(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.power


class Neg(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.negative


class Abs(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.absolute


class Matmul(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.matmul

    def _jacobian(self):
        ix = self.inputs["x"]
        iy = self.inputs["y"]
        M = ix.value.shape[0]
        N = iy.value.shape[1]
        K = ix.value.shape[1]
        assert K == iy.value.shape[0]
        grad_x = np.zeros((M * N, M * K))
        grad_y = np.zeros((M * N, N * K))
        for i in range(M):
            for j in range(N):
                grad_x[i * N + j, i * K:(i + 1) * K] = iy.value[:, j]
                grad_y[i * N + j, j + N * np.arange(K)] = ix.value[i, :]
        jac = {"x": grad_x, "y": grad_y}
        return jac
