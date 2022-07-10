import unittest

import numpy as np

from chell.core import tensor
from chell.core.optimizers import gd

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
        BATCH_SIZE = 3
        loss_v = 1
        count = 0
        while loss_v > 0.01:
            i = int(count % int(SAMPLE_COUNT / BATCH_SIZE))
            rows = list(range(BATCH_SIZE * i, BATCH_SIZE * (i + 1)))
            x.set_value(sample_x[:, rows])
            diff = model - sample_y[:, rows]
            diff_square = diff * diff
            loss = diff_square.sum()
            loss.forward()
            loss.backward()
            loss_v = loss.value[0]
            for p in params:
                grad = p.grad.mean(axis=0, keepdims=True)[0]
                p.value -= grad * 0.01
            count += 1
        print(f"test_direct_train finished with count:{count}")
        self.assertTrue(loss_v < 0.01)

    def test_optim_gd_train(self, momentum=0.0):
        model, x, params = self.create_model()
        optim = gd.GD(params, lr=0.01, momentum=momentum)
        BATCH_SIZE = 3
        loss_v = 1
        count = 0
        while loss_v > 0.01:
            i = int(count % int(SAMPLE_COUNT / BATCH_SIZE))
            rows = list(range(BATCH_SIZE * i, BATCH_SIZE * (i + 1)))
            x.set_value(sample_x[:, rows])
            diff = model - sample_y[:, rows]
            diff_square = diff * diff
            loss = diff_square.sum()
            loss.forward()
            loss.backward()
            loss_v = loss.value[0]
            optim.step()
            count += 1

        print(f"optim_gd_train lr={optim.lr} momentum={optim.momentum} finished with count:{count}")
        self.assertTrue(loss_v < 0.01)

    def test_optim_gd_train_with_momentum(self):
        self.test_optim_gd_train(momentum=0.8)


if __name__ == "__main__":
    unittest.main()
