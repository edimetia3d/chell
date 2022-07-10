import numpy as np

from chell.core import tensor

a = tensor.Tensor("a", np.ones((2, 3)), requires_grad=True)
x = tensor.Tensor("x", np.ones((3, 1)))
b = tensor.Tensor("b", np.ones((2, 1)), requires_grad=True)
model = a @ x + b

expect = tensor.Tensor("expect", np.ones((2, 1)))
diff = (model - expect)
diff_square = diff * diff
loss = diff_square.sum()

true_a = np.random.rand(2, 3)
true_b = np.random.rand(2, 1)
SAMPLE_COUNT = 1000
sample_x = np.random.randn(3, SAMPLE_COUNT)
sample_y = np.matmul(true_a, sample_x) + true_b

for i in range(SAMPLE_COUNT):
    x.set_value(sample_x[:, [i]])
    expect.set_value(sample_y[:, [i]])
    loss.forward()
    loss.backward()
    a.value = a.value - a.grad * 0.01
    b.value = b.value - b.grad * 0.01
print(a.value)
print(b.value)
assert (np.allclose(a.value, true_a) and np.allclose(b.value, true_b))
