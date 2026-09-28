"""The pipe protocol between gazebo_vec.py (host) and gazebo_env.py (container).

Length-prefixed pickles of plain Python only: the host's numpy (2.x) and the
container's (1.26) do not unpickle each other's arrays, so arrays travel as
(dtype, shape, bytes).
"""

from __future__ import annotations

import pickle
import struct

import numpy as np


def pack(x):
    """Plain Python for pickling across numpy versions (host 2.x, box 1.26)."""
    if isinstance(x, np.ndarray):
        return ("__nd__", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: pack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(pack(v) for v in x)
    return x


def unpack(x):
    if isinstance(x, tuple) and len(x) == 4 and x[0] == "__nd__":
        return np.frombuffer(x[3], dtype=np.dtype(x[1])).reshape(x[2]).copy()
    if isinstance(x, dict):
        return {k: unpack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(unpack(v) for v in x)
    return x


def send(stream, obj):
    data = pickle.dumps(pack(obj), protocol=4)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def receive(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError
    (size,) = struct.unpack("<Q", head)
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return unpack(pickle.loads(data))
