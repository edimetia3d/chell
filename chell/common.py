import hashlib
import os
from typing import Tuple

Shape = Tuple[int, ...]


def sha1(file_path: str, size_limit: int = 65536) -> str:
    if size_limit is None:
        buf_size = os.path.getsize(file_path)
    else:
        buf_size = min(size_limit, os.path.getsize(file_path))
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        data = f.read(buf_size)
        if not data:
            return None

        sha1.update(data)

    return sha1.hexdigest()
