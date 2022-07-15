import unittest

import numpy as np

from chell.core import tensor
from chell.core.losses import mse
from chell.core.optimizers import adagrad
from chell.core.optimizers import adam
from chell.core.optimizers import gd
from chell.core.optimizers import rmsprop

M = 2
N = 3
SAMPLE_COUNT = 1000

true_a = np.random.rand(M, N)
true_b = np.random.rand(M, 1)
sample_x = np.random.randn(N, SAMPLE_COUNT)
sample_y = np.matmul(true_a, sample_x) + true_b

EPSILON = 1e-6


class LinearRegressionTest(unittest.TestCase):

    def create_model(self):
        a = tensor.Tensor("a", np.ones((M, N)), requires_grad=True)
        x = tensor.Tensor("x", np.ones((N, 1)))
        b = tensor.Tensor("b", np.ones((M, 1)), requires_grad=True)
        return a @ x + b, x, [a, b]

    def launch_test(self, model, x, params, update_grad_fn, title, loss_fn=None):
        BATCH_SIZE = 3
        loss_v = 1
        count = 0

        def default_loss(output, label):
            diff = output - label
            diff_square = diff * diff
            return diff_square.sum()

        if loss_fn is None:
            loss_fn = default_loss

        while loss_v > EPSILON:
            i = int(count % int(SAMPLE_COUNT / BATCH_SIZE))
            rows = list(range(BATCH_SIZE * i, BATCH_SIZE * (i + 1)))
            x.set_value(sample_x[:, rows])
            loss = loss_fn(model, sample_y[:, rows])
            loss.forward()
            loss.backward()
            loss_v = loss.value[0]
            update_grad_fn()
            count += 1
        x = np.random.rand(N, 1)
        self.assertTrue(
            np.allclose(np.matmul(params[0].value, x) + params[1].value, np.matmul(true_a, x) + true_b, rtol=1e-2))
        print(f"{title} finished with count:{count}")

    def test_direct_train(self):
        model, x, params = self.create_model()

        def update_grad():
            for p in params:
                p.value -= p.grad * 0.01
                p.set_value(p.value)

        self.launch_test(model, x, params, update_grad, "test_direct_train")

    def test_optim_gd_train(self):
        model, x, (a, b) = self.create_model()
        optim = gd.GD(model.get_params(), lr=0.01)

        def update_grad():
            optim.step()

        self.launch_test(model, x, (a, b), update_grad, f"optim_gd_train lr={optim.lr} momentum={optim.momentum}")

    def test_optim_gd_train_with_momentum(self):
        model, x, params = self.create_model()
        optim = gd.GD(params, lr=0.01, momentum=0.8)

        def update_grad():
            optim.step()

        self.launch_test(model, x, params, update_grad, f"optim_gd_train lr={optim.lr} momentum={optim.momentum}")

    def test_optim_adagrad_train(self):
        model, x, params = self.create_model()
        optim = adagrad.AdaGrad(params, lr=1)

        def update_grad():
            optim.step()

        self.launch_test(model, x, params, update_grad, f"adagrad_train")

    def test_optim_rms_prop_train(self):
        model, x, params = self.create_model()
        optim = rmsprop.RMSProp(params, lr=0.01, beta=0.9)

        def update_grad():
            optim.step()

        self.launch_test(model, x, params, update_grad, f"rmsprop_train")

    def test_optim_adam_train(self):
        model, x, params = self.create_model()
        optim = adam.Adam(params, lr=0.01)

        def update_grad():
            optim.step()

        self.launch_test(model, x, params, update_grad, f"adam_train")

    def test_optim_adam_train_mse(self):
        model, x, params = self.create_model()
        optim = adam.Adam(params, lr=0.01)

        def update_grad():
            optim.step()

        self.launch_test(model, x, params, update_grad, f"adam_train_with_mse", mse.MSE)


if __name__ == "__main__":
    unittest.main()
