import numpy as np

from chell.core import op
from chell.core import tensor
from chell.core.ops import activation
from chell.core.ops import concat
from chell.core.ops import conv2d
from chell.core.ops import padding2d
from chell.core.ops import pooling2d
from chell.core.ops.composite import conv as composite_conv

shape = (2, 3)
x = tensor.Tensor("x", np.ones(shape), requires_grad=True)  # has value
y_with_v = tensor.Tensor("y", 2 * np.ones(shape))  # has no value
z = x + y_with_v  # eager execution, for all value is known
z2 = z * 3
assert (x + y_with_v).alleq(z)
assert ((x + y_with_v) * 3).alleq(z2)
print("==========")
z2.dump()
z2.backward()
print(x.grad)
y_no_v = tensor.Tensor()  # has no value
model = x * 2 + y_no_v  # no execution
assert model.value is None
y_no_v.value = np.random.random(shape)
model.forward()
assert (x * 2 + y_no_v).alleq(model)
print("==========")
model.dump()
model.backward()
print(x.grad)

print("==========")
rmat = tensor.Tensor("rmat", np.ones((shape[1], 4)))
mm = z2 @ rmat
mm.backward()
print(x.grad)

shapes = mm.shape_forward()
assert shapes[mm.node_name] == mm.value.shape
assert op.Operation.get_created_op_by_name(mm.node_name) is mm

a = tensor.Tensor("a", np.ones((5, 4)), requires_grad=True)
b = tensor.Tensor("b", np.ones((3, 4)))
c = tensor.Tensor("c", np.ones((8, 2)))

c0 = concat.Concat([a, b], axis=0)
c1 = concat.Concat([c0, c], axis=1)
c1.dump()
c1.forward()
c1.backward()

x = tensor.Tensor("x", np.arange(0, 3 * 3 * 3).reshape((3, 3, 3)))
kernel = tensor.Tensor("kernel", np.arange(0, 3 * 2 * 2).reshape((3, 2, 2)), requires_grad=True)
conv = conv2d.Conv2D(x, kernel)
conv.backward()

xs = [tensor.Tensor("x", np.ones((3, 3, 3)), requires_grad=True) for i in range(4)]
cps_cv = composite_conv.conv(xs, (3, 3, 3), 2, 2, 2, 2, True, activation.LeakyRelu)
cps_cv.dump()
cps_cv.backward()

x = tensor.Tensor("x", np.arange(0, 3 * 5 * 5).reshape((3, 5, 5)), requires_grad=True)
pool = pooling2d.Pooling2D(x, pooling2d.PollingMode.MAX, (2, 2), (2, 2))
pool.backward()

x = tensor.Tensor("x", np.arange(0, 2 * 2 * 2).reshape((2, 2, 2)), requires_grad=True)
pad_config = padding2d.PaddingConfig(padding_size=(1, 2, 3, 4), padding_mode=padding2d.PaddingMode.CLAMP_TO_CONST,
                                     clamp_value=0)
pad = padding2d.Padding2D(x, pad_config)
pad.backward()
