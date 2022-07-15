# all operation in this file will be used in the op.Operation class

from typing import Callable, Any, Union, Dict, List, Tuple

import numpy as np

from chell import common
from chell.core import op


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

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        # treat all Binary OP as elementwise by default
        return input_shapes["x"]

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        # treat all Binary OP as elementwise by default
        if shape_var_list is None:
            return 2
        else:
            return {"x": tuple(shape_var_list), "y": tuple(shape_var_list)}


def _eltwise_jac(op: op.Operation,
                 base_jac: Callable[[np.ndarray, np.ndarray, int], Tuple[int, int]]):
    ix = op.inputs["x"].value
    Nx = ix.size
    indx = np.arange(Nx).reshape(ix.shape)
    iy = op.inputs["y"].value
    Ny = iy.size
    indy = np.arange(Ny).reshape(iy.shape)
    iz = op.value
    Nz = iz.size
    indx = np.broadcast_to(indx, iz.shape)
    indy = np.broadcast_to(indy, iz.shape)
    ix = np.broadcast_to(ix, iz.shape)
    iy = np.broadcast_to(iy, iz.shape)

    jacx = np.zeros((Nz, Nx))
    jacy = np.zeros((Nz, Ny))
    for i in range(Nz):
        x_ind = indx.ravel()[i]
        y_ind = indy.ravel()[i]
        jx, jy = base_jac(ix.ravel()[i], iy.ravel()[i])
        jacx[i, x_ind] = jx
        jacy[i, y_ind] = jy

    return {"x": jacx, "y": jacy}


class Add(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.add

    def _jacobian(self):
        def base_jac(vx, vy):
            return 1, 1

        return _eltwise_jac(self, base_jac)


class Sub(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.subtract

    def _jacobian(self):
        def base_jac(vx, vy):
            return 1, -1

        return _eltwise_jac(self, base_jac)


class Mul(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.multiply

    def _jacobian(self):
        def base_jac(vx, vy):
            return vy, vx

        return _eltwise_jac(self, base_jac)


class Div(_BinaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.divide

    def _jacobian(self):
        def base_jac(vx, vy):
            return 1 / vy, -1 * vx / vy ** 2

        return _eltwise_jac(self, base_jac)


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

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        x_shape = input_shapes["x"]
        y_shape = input_shapes["y"]
        M = x_shape[0]
        N = y_shape[1]
        return (M, N)

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 3
        else:
            return {"x": (shape_var_list[0], shape_var_list[1]), "y": (shape_var_list[1], shape_var_list[2])}


class _UnaryOp(op.Operation):
    _unary_np_func: Callable[[Any], np.ndarray] = None
    _OP_NAME: str = None

    def __init__(self, x: op.OpArgT, ):
        op_name = self._OP_NAME
        if op_name is None:
            op_name = self.__class__.__name__.lower()
        super().__init__(op_name, {"x": x})

    def _compute(self):
        ix = self.inputs["x"]
        return self.__class__._unary_np_func(ix.value)

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        return input_shapes["x"]

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 0
        else:
            return {"x": tuple(shape_var_list)}


class Neg(_UnaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.negative


class Reciprocal(_UnaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.reciprocal


class Transpose(_UnaryOp):
    _binary_np_func: Callable[[Any, Any], np.ndarray] = np.transpose

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 2
        else:
            return {"x": (shape_var_list[0], shape_var_list[1])}

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        return input_shapes["x"][::-1]


class _Reduce(op.Operation):
    _reduce_np_func: Callable[..., np.ndarray] = None
    _OP_NAME: str = None

    def __init__(self, x: op.OpArgT, axis: Union[int, None]):
        op_name = self._OP_NAME
        if op_name is None:
            op_name = self.__class__.__name__.lower()
        self.axis = axis
        super().__init__(op_name, {"x": x})

    def _compute(self):
        ix = self.inputs["x"]
        v = self.__class__._reduce_np_func(ix.value, self.axis, keepdims=True)
        if np.isscalar(v):
            v = np.array([v])
        return v

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 0
        else:
            return {"x": tuple(shape_var_list)}

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        x_shape = input_shapes["x"]
        if self.axis is None:
            return (1,)
        else:
            return tuple(x_shape[:self.axis] + [1] + x_shape[self.axis + 1:])


class ReduceSum(_Reduce):
    _reduce_np_func = np.sum

    def _jacobian(self):
        ix = self.inputs["x"]
        jac = {}
        jac["x"] = np.ones(shape=(1, ix.value.size))
        return jac
