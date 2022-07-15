import unittest
from typing import List, Tuple

import numpy as np

from chell import data_loader
from chell.core import op
from chell.core import tensor
from chell.core.losses import softmax_cross_entropy
from chell.core.ops import activation
from chell.core.ops import fc
from chell.core.optimizers import adam
from chell.datasets import mnist


class LinearRegressionTest(unittest.TestCase):

    def create_model(self, hidden_size: List[int] = None) -> Tuple[op.Operation, tensor.Tensor]:
        if hidden_size is None:
            hidden_size = [10]

        input0 = tensor.Tensor("flag_img", np.ones((28 * 28, 1)))
        act = input0 / 255 - 0.5
        m_w = 28 * 28
        m_x = None
        for hs in hidden_size:
            m_x = m_w
            m_w = hs
            act = fc.FullyConnect(m_w, m_x, 1, act, activation.LeakyRelu)
        m_x = m_w
        m_w = 10
        return fc.FullyConnect(m_w, m_x, 1, act, activation.LeakyRelu), input0

    def test_optim_adam_train(self):
        model, flat_img = self.create_model([300, 100])
        optim = adam.Adam(model.get_params(), lr=0.01)
        train_set = mnist.MNISTTrain()
        test_set = mnist.MNISTTest()
        label_tensor = tensor.Tensor("label", np.ones((10, 1)))
        loss = softmax_cross_entropy.SoftMaxCrossEntropy(model, label_tensor)
        EPOCH = 1
        BATCHSIZE = 4
        self.launch_model(test_set, flat_img, model, "initial result")
        print("\n")
        train_part, eval_part = train_set.split(0.9, shuffle=True)
        continue_passed = 0
        for epoch in range(EPOCH):
            img_id = 0
            for raw_img, label in data_loader.DataLoader(train_part, batch_size=BATCHSIZE):
                raw_img = [img.reshape((28 * 28, 1)) for img in raw_img]
                img = np.concatenate(raw_img, axis=1)
                flat_img.set_value(img)
                label_one_hot = np.zeros((10, BATCHSIZE))
                for i, l in enumerate(label):
                    label_one_hot[l, i] = 1
                label_tensor.set_value(label_one_hot)
                loss.forward()
                loss.backward()
                loss_v = loss.value[0]
                if img_id % 100 == 0:
                    print(f"epoch {epoch} img_id {img_id} loss:{loss_v:g} ", end="")
                    acc = self.launch_model(eval_part, flat_img, model, "eval result")
                    if acc > 0.9:
                        continue_passed += 1
                    else:
                        continue_passed = 0
                    if continue_passed >= 10:
                        break
                optim.step()
                img_id += 1
        print("\n")

        self.launch_model(test_set, flat_img, model, "final result")

    def launch_model(self, dset, flat_img, model, title):
        soft_max_out = activation.SoftMax(model)
        correct_count = 0
        count = len(dset)
        for img, label in data_loader.DataLoader(dset, batch_size=1):
            img = img[0]
            flat_img.set_value(img.reshape((28 * 28, 1)))
            soft_max_out.forward()
            ind = np.argmax(soft_max_out.value, axis=0)
            if ind == label[0][0]:
                correct_count = correct_count + 1
        print(f"{title} correct_count:{correct_count} count:{count}")
        return correct_count / count


if __name__ == "__main__":
    unittest.main()
