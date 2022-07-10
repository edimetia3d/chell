import unittest

import numpy as np

from chell.core import tensor

M = 2
N = 3
SAMPLE_COUNT = 1000

true_a = np.random.rand(M, N)
true_b = np.random.rand(M, 1)
sample_x = np.random.randn(N, SAMPLE_COUNT)
sample_y = np.matmul(true_a, sample_x) + true_b


class LinearRegressionTest(unittest.TestCase):
    def create_model(self):
        a = tensor.Tensor("a", np.ones((M, N)), requires_grad=True)
        x = tensor.Tensor("x", np.ones((N, 1)))
        b = tensor.Tensor("b", np.ones((M, 1)), requires_grad=True)
        return a @ x + b, x, [a, b]

    def test_direct_train(self):
        model, x, params = self.create_model()
        for i in range(SAMPLE_COUNT):
            x.set_value(sample_x[:, [i]])
            diff = model - sample_y[:, [i]]
            diff_square = diff * diff
            loss = diff_square.sum()
            loss.forward()
            loss.backward()
            for p in params:
                p.value -= p.grad * 0.01

        self.assertTrue(np.allclose(params[0].value, true_a) and np.allclose(params[1].value, true_b))
        self.assertTrue(np.allclose(loss.value, 0))


if __name__ == "__main__":
    unittest.main()
