def uuid7_from_values(unix_ts_ms: int, rand_a: int, rand_b: int) -> bytes:
    version = 0x07
    var = 2
    rand_a &= 0xFFF
    rand_b &= 0x3FFFFFFFFFFFFFFF

    final_bytes = unix_ts_ms.to_bytes(6, byteorder="big")
    final_bytes += ((version << 12) + rand_a).to_bytes(2, "big")
    final_bytes += ((var << 62) + rand_b).to_bytes(8, "big")

    return final_bytes
