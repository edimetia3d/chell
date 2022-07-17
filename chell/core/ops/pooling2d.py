import enum
from typing import Tuple, Dict

import numpy as np

from chell.core import op


class PollingMode(enum.Enum):
    """Same as OPENGL's definition."""
    MAX = 0


class _Polling(op.Operation):

    def __init__(self,
                 x: op.OpArgT,
                 ksize: Tuple[int, int],
                 stride: Tuple[int, int]):
        """x is a 3D tensor of shape (C, H, W)"""
        self.ksize = ksize
        self.stride = stride
        super().__init__("conv2d", {"x": x})

    def _kerner_shape(self):
        return (self.inputs["x"].value.shape[0],) + tuple(self.ksize)

    def _shape_without_stride(self):
        x = self.inputs["x"].value
        kernel_shape = self._kerner_shape()
        return tuple(np.subtract(x.shape, kernel_shape) + 1)

    def _x_stride_ratio(self):
        return (1, self.stride[0], self.stride[1])

    def _build_view_matrix(self):
        x = self.inputs["x"].value
        kernel_shape = self._kerner_shape()
        kernel_size = np.prod(kernel_shape)
        final_shape = tuple(
            np.ceil(np.divide(self._shape_without_stride(), self._x_stride_ratio())).astype(dtype=np.int))
        view_shape = final_shape + kernel_shape
        strides = tuple(np.multiply(x.strides, self._x_stride_ratio())) + x.strides

        sub_matrices = np.lib.stride_tricks.as_strided(x, view_shape, strides)
        x_matrix_view = sub_matrices.reshape((-1, kernel_size))
        return x_matrix_view, final_shape


class MaxPolling(_Polling):

    def _compute(self) -> np.ndarray:
        x_matrix_view, final_shape = self._build_view_matrix()
        max_v_j_ind = np.argmax(x_matrix_view, axis=1)
        self._max_v_kernel_offset = max_v_j_ind
        return np.max(x_matrix_view, axis=1).reshape(final_shape)

    def _jacobian(self) -> Dict[str, np.ndarray]:
        x = self.inputs["x"].value
        final_shape = self.value.shape
        kernel_shape = self._kerner_shape()
        jac_x = np.zeros((self.value.size, x.size))
        for row, _ind in enumerate(np.ndindex(*final_shape)):
            ind = tuple(np.multiply(_ind, self._x_stride_ratio()).astype(dtype=np.int))
            kernel_offset = self._max_v_kernel_offset[row]
            ind_in_kernel = np.unravel_index([kernel_offset], kernel_shape)
            real_ind = np.add(ind, next(zip(*ind_in_kernel)))
            multi_x_ind = [[x] for x in real_ind]
            offset = np.ravel_multi_index(multi_x_ind, x.shape)
            jac_x[row, offset] = 1
        return {"x": jac_x}


def Pooling2D(x: op.OpArgT,
              mode: PollingMode = PollingMode.MAX,
              ksize: Tuple[int, int] = (2, 2),
              stride: Tuple[int, int] = (2, 2)) -> op.Operation:
    if mode == PollingMode.MAX:
        return MaxPolling(x, ksize, stride)
