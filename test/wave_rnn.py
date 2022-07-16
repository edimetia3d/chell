import numpy as np

from chell import data_loader
from chell import dataset
from chell.core import tensor
from chell.core.losses import softmax_cross_entropy
from chell.core.ops import activation
from chell.core.ops import fc
from chell.core.optimizers import adam


class WaveDataset(dataset.DataSet):
    def __init__(self, dimension=10, x_num=10, len: int = 1000):
        self.datas = []
        self.labels = []
        for j in range(len):
            xv = np.arange(0, 10, 10 / x_num)
            i = j % 2
            if i == 0:
                wave = np.sin(xv)
                label = np.array([1, 0]).reshape(2, 1)
            else:
                wave = self.square_wave(xv)
                label = np.array([0, 1]).reshape(2, 1)
            self.labels.append(label)
            sequence = wave + np.random.normal(0, 0.6, (dimension, x_num)) / 3
            self.datas.append(sequence)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item], self.labels[item]

    def square_wave(self, x: np.ndarray, duty: float = 0.5) -> np.ndarray:
        T0 = 2 * np.pi * duty
        T = 2 * np.pi
        mod_x = x - (np.floor_divide(x, T) * T)
        y = np.ones(mod_x.shape)
        y[mod_x > T0] = -1
        return y


seq_len = 50  # num of input
dimension = 10  # dimension of single input
status_dimension = 20


def build_model():
    inputs = [tensor.Tensor(f"x{i}", np.ones(shape=(dimension, 1))) for i in range(seq_len)]
    U = tensor.Tensor("U", np.random.randn(status_dimension, dimension), requires_grad=True)
    W = tensor.Tensor("W", np.random.randn(status_dimension, status_dimension), requires_grad=True)
    b = tensor.Tensor("b", np.random.randn(status_dimension, 1), requires_grad=True)

    last_step = None
    for iv in inputs:
        h = U @ iv + b
        if last_step is not None:
            h = W @ last_step + h

        last_step = activation.LeakyRelu(h)

    fc1 = fc.FullyConnect(40, status_dimension, 1, last_step, activation.LeakyRelu)
    fc2 = fc.FullyConnect(10, 40, 1, fc1, activation.LeakyRelu)
    output = fc.FullyConnect(2, 10, 1, fc2, None)

    label = tensor.Tensor("label", np.ones(shape=(2, 1)))
    loss = softmax_cross_entropy.SoftMaxCrossEntropy(output, label)

    predict = activation.SoftMax(output)
    loss.dump()
    return inputs, predict, label, loss


def eval_accuracy(title, predict, inputs, dset):
    correct = 0
    count = 0
    for wave, label in data_loader.DataLoader(dset, batch_size=1):
        count += 1
        for j, x in enumerate(inputs):
            x.set_value(wave[0][:, [j]])

        predict.forward()
        ind = np.argmax(predict.value, axis=0)
        if label[0].ravel()[ind[0]] == 1:
            correct += 1

    print(f"{title} eval accuracy: {100 * correct / count:.2f} %")

    return correct / count


def train():
    wave_set = WaveDataset(dimension=dimension, x_num=seq_len)
    inputs, predict, label, loss = build_model()
    optimizer = adam.Adam(predict.get_params(), 0.005)

    train_eval_set, test_set = wave_set.split(0.8, shuffle=True)
    train_set, eval_set = train_eval_set.split(0.9)

    EPOCH = 50
    coninue_good = 0
    for epoch in range(EPOCH):
        iter_count = 0
        for wave, l in data_loader.DataLoader(train_set, batch_size=1, shuffle=True):
            for j, x in enumerate(inputs):
                x.set_value(wave[0][:, [j]])

            label.set_value(l[0])

            loss.forward()
            loss_v = loss.value[0]
            loss.backward()
            optimizer.step()
            if iter_count % 100 == 0:
                acc = eval_accuracy(f"epoch {epoch} iter {iter_count} loss:{loss_v:g}", predict, inputs, eval_set)
                if acc > 0.95:
                    coninue_good += 1
                else:
                    coninue_good = 0
                if coninue_good > 10:
                    eval_accuracy("Final", predict, inputs, test_set)
                    return

            iter_count += 1


if __name__ == "__main__":
    train()
