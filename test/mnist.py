import unittest
from typing import List
from typing import Tuple

import numpy as np

from chell import data_loader
from chell.core import op
from chell.core import tensor
from chell.core.losses import softmax_cross_entropy
from chell.core.ops import activation
from chell.core.ops import fc
from chell.core.ops import padding2d
from chell.core.ops import pooling2d
from chell.core.ops import reshape
from chell.core.ops.composite import conv
from chell.core.optimizers import adam
from chell.datasets import mnist


class FullyConnectTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BATCHSIZE = 4
        self.EPOCH = 1
        self.lr = 0.01

    def build_model(self, hidden_size: List[int]) -> Tuple[op.Operation, tensor.Tensor]:
        input0 = tensor.Tensor("input", np.ones((1, 28, 28)))
        reshaped = reshape.Reshape(input0, np.array((28 * 28, -1)))
        act = reshaped / 255 - 0.5
        m_w = 28 * 28
        m_x = None
        for hs in hidden_size:
            m_x = m_w
            m_w = hs
            act = fc.FullyConnect(m_w, m_x, 1, act, activation.LeakyRelu)
        m_x = m_w
        m_w = 10
        return fc.FullyConnect(m_w, m_x, 1, act, None), input0

    def test_optim_adam_train(self):
        model, input0 = self.build_model([300, 100])
        model.dump()
        optim = adam.Adam(model.get_params(), lr=self.lr)

        train_eval_set = mnist.MNISTTrain()
        test_set = mnist.MNISTTest()
        train_set, eval_set = train_eval_set.split(0.9, shuffle=True)

        label_tensor = tensor.Tensor("label", np.ones((10, 1)))
        loss = softmax_cross_entropy.SoftMaxCrossEntropy(model, label_tensor)

        predict = activation.SoftMax(model)

        continue_passed = 0
        for epoch in range(self.EPOCH):
            batch_id = 0
            for raw_img, label in data_loader.DataLoader(train_set, batch_size=self.BATCHSIZE):
                raw_img = [img.reshape((28 * 28, 1)) for img in raw_img]
                img = np.concatenate(raw_img, axis=1)
                input0.set_value(img)
                label_one_hot = np.zeros((10, self.BATCHSIZE))
                for i, l in enumerate(label):
                    label_one_hot[l, i] = 1
                label_tensor.set_value(label_one_hot)
                loss.forward()
                loss.backward()
                loss_v = loss.value[0]
                if batch_id % 100 == 0:
                    acc = self.launch_model(eval_set, input0, predict)
                    if acc > 0.9:
                        continue_passed += 1
                    else:
                        continue_passed = 0
                    if continue_passed >= 10:
                        break
                    print(f"epoch {epoch} batch_id {batch_id} loss:{loss_v:g} acc:{acc:g}")
                optim.step()
                batch_id += 1

        acc = self.launch_model(test_set, input0, model)
        print(f"Final acc:{acc:g}")

    def launch_model(self, dset, input0, predict):
        correct_count = 0
        for raw_img, label in data_loader.DataLoader(dset, batch_size=1):
            raw_img = [img.reshape((28 * 28, 1)) for img in raw_img]
            img = np.concatenate(raw_img, axis=1)
            input0.set_value(img)
            predict.forward()
            ind = np.argmax(predict.value, axis=0)
            if ind == label[0][0]:
                correct_count = correct_count + 1
        return correct_count / len(dset)


class CNNTest(FullyConnectTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BATCHSIZE = 1
        self.EPOCH = 1
        self.lr = 0.01

    def build_model(*args, **kwargs) -> Tuple[op.Operation, tensor.Tensor]:
        i_shape = (1, 28, 28)
        raw = tensor.Tensor("flat_img", np.ones(i_shape))
        input0 = reshape.Reshape(raw, np.array(i_shape))
        act = input0 / 255 - 0.5
        pad_config0 = padding2d.PaddingConfig(padding_size=(2, 2, 2, 2),
                                              padding_mode=padding2d.PaddingMode.CLAMP_TO_CONST,
                                              clamp_value=0)
        conv1 = conv.conv([act], i_shape, 5, 5, 3, padding_config=pad_config0, Act=activation.LeakyRelu)

        pad_config1 = padding2d.PaddingConfig(padding_size=(1, 1, 1, 1),
                                              padding_mode=padding2d.PaddingMode.CLAMP_TO_CONST,
                                              clamp_value=0)
        conv1_pad = padding2d.Padding2D(conv1, pad_config1)
        polling1 = pooling2d.Pooling2D(conv1_pad, pooling2d.PollingMode.MAX, (3, 3), (2, 2))

        conv2 = conv.conv([polling1], (3, 14, 14), 3, 3, 3, padding_config=pad_config1, Act=activation.LeakyRelu)
        conv2_pad = padding2d.Padding2D(conv2, pad_config1)
        pooling2 = pooling2d.Pooling2D(conv2_pad, pooling2d.PollingMode.MAX, (3, 3), (2, 2))
        flatten = reshape.Reshape(pooling2, np.array((7 * 7 * 3, -1)))
        fc1 = fc.FullyConnect(120, 7 * 7 * 3, 1, flatten, activation.LeakyRelu)
        output = fc.FullyConnect(10, 120, 1, fc1, None)
        return output, raw


if __name__ == "__main__":
    unittest.main()
