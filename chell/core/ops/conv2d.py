from typing import Tuple, Dict

import numpy as np

from chell.core import op
from chell.core.ops import padding2d


class Conv2D(op.Operation):

    def __init__(self,
                 x: op.OpArgT,
                 kernel: op.OpArgT,
                 stride: Tuple[int, int] = (1, 1),
                 padding_config: padding2d.PaddingConfig = None,
                 dilation: Tuple[int, int] = (0, 0)):
        """ Do convolution on x by kernel.

        x is a 3D tensor of shape (IC, H, W), kernel is a 3D tensor of shape (IC, KH, KW)

        Stride is defines how kernel moves to next position.

        Padding is a pre-processing on the `x`, we will extend x from (IC, H, W) to (IC, H + PadH_L+PadH_R, W+PadW_U+PadW_D)

        Dilation is a pre-processing on the `kernel`, we will extend kernel from (KC, KH, KW) to (KC, KH*(DilH+1), KW*(DilW+1)),
        by injecting zeros between the original kernel pixel



        Args:
            x: 3D tensor of shape (IC, H, W)
            kernel: 3D tensor of shape (IC, KH, KW)
            padding_config: config how the input `x` is padded, default is None, which means no padding.
            stride: stride of the convolution, default is (1, 1)
            dilation: dilation of the convolution, default is (0, 0)
        """
        self.stride = stride
        self.dilation = dilation
        if padding_config is not None:
            x = padding2d.Padding2D(x, padding_config)

        super().__init__("conv2d", {"x": x, "kernel": kernel})

    def _compute(self):
        assert self.stride == (1, 1), "stride is not implemented yet"
        assert self.dilation == (0, 0), "dilation is not implemented yet"
        x = self.inputs["x"].value
        kernel = self.inputs["kernel"].value
        final_shape = tuple(np.subtract(x.shape, kernel.shape) + 1)
        view_shape = final_shape + kernel.shape
        strides = x.strides + x.strides

        sub_matrices = np.lib.stride_tricks.as_strided(x, view_shape, strides)
        x_matrix_view = sub_matrices.reshape((-1, kernel.size))
        kernel_matrix_view = kernel.reshape((kernel.size, 1))
        conv = np.matmul(x_matrix_view, kernel_matrix_view).reshape(final_shape)
        return conv

    def _jacobian(self) -> Dict[str, np.ndarray]:
        x = self.inputs["x"].value
        kernel = self.inputs["kernel"].value
        final_shape = self.value.shape
        jac_x = np.zeros((self.value.size, x.size))
        jac_kernel = np.zeros((self.value.size, kernel.size))
        for row, ind in enumerate(np.ndindex(*final_shape)):
            for offset_kernel, ind_kernel in enumerate(np.ndindex(*kernel.shape)):
                x_ind = tuple(np.add(ind, ind_kernel))
                jac_kernel[row, offset_kernel] = x[x_ind]
                multi_x_ind = [[x] for x in x_ind]
                offset = np.ravel_multi_index(multi_x_ind, x.shape)
                jac_x[row, offset[0]] = kernel[ind_kernel]
        return {"x": jac_x, "kernel": jac_kernel}
