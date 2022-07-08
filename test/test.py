import numpy as np

from chell.core import tensor

shape = (2, 3)
x = tensor.Tensor("x", np.ones(shape), requires_grad=True)  # has value
y_with_v = tensor.Tensor("y", 2 * np.ones(shape))  # has no value
z = x + y_with_v  # eager execution, for all value is known
z2 = z * 3
assert x + y_with_v == z
assert (x + y_with_v) * 3 == z2
print("==========")
z2.dump()
z2.backward()
print(x.grad)
y_no_v = tensor.Tensor()  # has no value
model = x * 2 + y_no_v  # no execution
assert model.value is None
y_no_v.value = np.random.random(shape)
model.forward()
assert x * 2 + y_no_v == model
print("==========")
model.dump()
model.backward()
print(x.grad)

print("==========")
rmat = tensor.Tensor("rmat", np.ones((shape[1], 4)))
mm = z2 @ rmat
mm.backward()
print(x.grad)
