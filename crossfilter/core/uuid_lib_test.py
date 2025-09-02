from datetime import datetime
from crossfilter.core import uuid_lib
import uuid


def test_uuid7_from_values() -> None:
    d = datetime(2024, 1, 1, 0, 0, 0, 0)
    milliseconds = int(d.timestamp() * 1e3)

    uuid_bytes = uuid_lib.uuid7_from_values(milliseconds, 0, 0)
    uuid_int = int.from_bytes(uuid_bytes, byteorder="big")
    uuid7 = uuid.UUID(int=uuid_int)
