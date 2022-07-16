import numpy as np

from chell.core import op
from chell.core import tensor
from chell.core.ops import concat

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
