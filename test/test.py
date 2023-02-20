import numpy as np
import torch

from chell.core import op
from chell.core import tensor
from chell.core.ops import activation
from chell.core.ops import concat
from chell.core.ops import conv2d
from chell.core.ops import padding2d
from chell.core.ops import pooling2d
from chell.core.ops import reshape
from chell.core.ops.composite import conv as composite_conv

shape = (2, 3)
x = tensor.Tensor("x", np.ones(shape), requires_grad=True)  # has value
y_with_v = tensor.Tensor("y", 2 * np.ones(shape))  # has no value
z = x + y_with_v  # eager execution, for all value is known
z2 = z * 3
assert (x + y_with_v).value_eq(z)
assert ((x + y_with_v) * 3).value_eq(z2)
print("==========")
z2.dump()
z2.backward()
print(x.grad)
y_no_v = tensor.Tensor()  # has no value
model = x * 2 + y_no_v  # no execution
assert model.value is None
y_no_v.value = np.random.random(shape)
model.forward()
assert (x * 2 + y_no_v).value_eq(model)
print("==========")
model.dump()
model.backward()
print(x.grad)

print("==========")
rmat = tensor.Tensor("rmat", np.ones((shape[1], 4)))
mm = z2 @ rmat
mm.backward()
print(x.grad)


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
cps_cv = composite_conv.conv(xs, (3, 3, 3), 2, 2, 2, 2, add_bias=True, Act=activation.LeakyRelu)
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

a = tensor.Tensor("a", np.ones((5, 4)), requires_grad=True)
a_r = reshape.Reshape(a, np.array((1, 4, 5)))
a_r.backward()

x = tensor.Tensor("x", np.random.random((3, 5, 6)), requires_grad=True)
kernel = tensor.Tensor("kernel", np.random.random((3, 2, 2)), requires_grad=True)
c = conv2d.Conv2D(x, kernel)
c.backward()

xt = torch.from_numpy(x.value.reshape((1, 3, 5, 6)))
xt.requires_grad = True
kernelt = torch.from_numpy(kernel.value.reshape((1, 3, 2, 2)))
kernelt.requires_grad = True
ct = torch.nn.functional.conv2d(xt, kernelt)
ct.backward(torch.ones(ct.shape))

assert np.allclose(c.value, ct.detach().numpy())
assert np.allclose(x.grad, xt.grad.detach().numpy())
assert np.allclose(kernel.grad, kernelt.grad.detach().numpy())

mp = pooling2d.Pooling2D(x, pooling2d.PollingMode.MAX, (2, 2), (2, 2))
mp.backward()
mpt = torch.nn.functional.max_pool2d(xt, kernel_size=(2, 2), stride=(2, 2))
xt.grad = None
mpt.backward(torch.ones(mpt.shape))

assert (np.allclose(mp.value, mpt.detach().numpy()))
assert np.allclose(x.grad, xt.grad)

pad_config = padding2d.PaddingConfig(padding_size=(1, 2, 3, 4), padding_mode=padding2d.PaddingMode.CLAMP_TO_CONST,
                                     clamp_value=0)
pd = padding2d.Padding2D(x, pad_config)
pd.backward()
pdt = torch.nn.functional.pad(xt, pad=(1, 2, 3, 4), mode='constant', value=0)
xt.grad = None
pdt.backward(torch.ones(pdt.shape))

assert np.allclose(pd.value, pdt.detach().numpy())
assert np.allclose(x.grad, xt.grad)

xs = [tensor.Tensor("x", np.random.randn(3, 4, 5), requires_grad=True) for i in range(4)]
xst = [torch.from_numpy(x.value) for x in xs]
for x in xst:
    x.requires_grad = True

cc = concat.Concat(xs, axis=1)
cct = torch.cat(xst, dim=1)
cct.backward(torch.ones(cct.shape))
cc.backward()

assert np.allclose(cc.value, cct.detach().numpy())
for i in range(4):
    assert np.allclose(xs[i].grad, xst[i].grad)

x = tensor.Tensor("x", np.random.randn(3, 4, 5), requires_grad=True)
xt = torch.from_numpy(x.value)
xt.requires_grad = True
x_r = reshape.Reshape(x, np.array((3, 20)))
xt_r = torch.reshape(xt, (3, 20))
x_r.backward()
xt_r.backward(torch.ones(xt_r.shape))

assert np.allclose(x_r.value, xt_r.detach().numpy())
assert np.allclose(x.grad, xt.grad)
