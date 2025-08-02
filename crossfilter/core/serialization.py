import numpy as np
import msgpack_numpy

msgpack_numpy.patch()


def serialize_numpy_array(array: np.ndarray) -> bytes:
    result = msgpack_numpy.packb(array)
    assert isinstance(result, bytes)
    return result


def deserialize_numpy_array(blob: bytes) -> np.ndarray:
    result = msgpack_numpy.unpackb(blob)
    assert isinstance(result, np.ndarray)
    return result
