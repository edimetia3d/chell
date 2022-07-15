import gzip
import os
import shutil
import urllib.request

import numpy as np

from chell import common
from chell import dataset

_TRAIN_SET = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
_TRAIN_SET_HASH = "e1f0426829a11cbe2a9c44ac744c36910c47c7aa"
_TRAIN_LABEL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
_TRAIN_LABEL_HASH = "adbf52269f5d842899f287c269e2883e40b4f6e2"
_TEST_SET = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
_TEST_SET_HASH = "2b3b5070bd24f613ab7ef5507b60065a32eaf4aa"
_TEST_LABEL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
_TEST_LABEL_HASH = "a6d52cc628797e845885543326e9f10abb8a6f89"


# IMAGE FORMAT
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel in row0 col0
# 0017     unsigned byte   ??               pixel in row0 col1
# ...

# LABEL FORMAT
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label

def _show_link_progress(link):
    def _show_progress(block_num, block_size, total_size):
        print(f"\rDownloading {link}: %.2f%%" % (block_num * block_size * 100 / total_size), end="")

    return _show_progress


# extracting file

class MNIST(dataset.DataSet):

    def __init__(self, cache_dir):
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.raw_img = None
        self.raw_label = None

    def _update_cache(self, download_link: str, gz_file: str, hash: str):
        final_file = os.path.join(self.cache_dir, os.path.basename(gz_file).split(".")[0])
        if not os.path.exists(gz_file) or hash != common.sha1(final_file):
            urllib.request.urlretrieve(download_link, gz_file, _show_link_progress(download_link))
            with gzip.open(gz_file, 'r') as f_in:
                with open(final_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def load_raw_data(self, binary_img: bytes, binary_label: bytes):
        binary_img = np.frombuffer(binary_img, dtype=np.uint8)
        u32_img_view = binary_img[0:16].view(dtype=">i4")
        assert u32_img_view[0] == 2051, "Invalid image file"
        img_num = u32_img_view[1]
        img_row = u32_img_view[2]
        img_col = u32_img_view[3]
        self.raw_img = binary_img[16:].reshape(img_num, img_row, img_col)

        binary_label = np.frombuffer(binary_label, dtype=np.uint8)
        u32_label_view = binary_label.view(dtype=">i4")
        assert u32_label_view[0] == 2049, "Invalid label file"
        label_num = u32_label_view[1]
        self.raw_label = binary_label[8:].reshape(label_num, 1)

    def __len__(self):
        return self.raw_img.shape[0]

    def __getitem__(self, item):
        return self.raw_img[item], self.raw_label[item]


class MNISTTrain(MNIST):

    def __init__(self, cache_dir="~/.cache/chell/dataset/mnist"):
        super().__init__(cache_dir)

        self._update_cache(_TRAIN_SET, os.path.join(self.cache_dir, "train-images-idx3-ubyte.gz"), _TRAIN_SET_HASH)
        self._update_cache(_TRAIN_LABEL, os.path.join(self.cache_dir, "train-labels-idx1-ubyte.gz"), _TRAIN_LABEL_HASH)
        with open(os.path.join(self.cache_dir, "train-images-idx3-ubyte"), 'rb') as f:
            with open(os.path.join(self.cache_dir, "train-labels-idx1-ubyte"), 'rb') as f_label:
                self.load_raw_data(f.read(), f_label.read())


class MNISTTest(MNIST):

    def __init__(self, cache_dir="~/.cache/chell/dataset/mnist"):
        super().__init__(cache_dir)

        self._update_cache(_TEST_SET, os.path.join(self.cache_dir, "t10k-images-idx3-ubyte.gz"), _TEST_SET_HASH)
        self._update_cache(_TEST_LABEL, os.path.join(self.cache_dir, "t10k-labels-idx1-ubyte.gz"), _TEST_LABEL_HASH)
        with open(os.path.join(self.cache_dir, "t10k-images-idx3-ubyte"), 'rb') as f:
            with open(os.path.join(self.cache_dir, "t10k-labels-idx1-ubyte"), 'rb') as f_label:
                self.load_raw_data(f.read(), f_label.read())
