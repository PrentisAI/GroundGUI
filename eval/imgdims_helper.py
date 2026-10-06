"""Read image width/height from the PNG/JPEG file header without decoding pixels."""
import struct


def dims(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(26)
            if head[:8] == b'\x89PNG\r\n\x1a\n':
                w, h = struct.unpack('>II', head[16:24])
                return int(w), int(h)
            if head[:2] == b'\xff\xd8':
                f.seek(2)
                while True:
                    b = f.read(1)
                    if not b:
                        return None
                    if b != b'\xff':
                        continue
                    while b == b'\xff':
                        b = f.read(1)
                    m = b[0]
                    if 0xc0 <= m <= 0xcf and m not in (0xc4, 0xc8, 0xcc):
                        f.read(3)
                        h, w = struct.unpack('>HH', f.read(4))
                        return int(w), int(h)
                    ln = struct.unpack('>H', f.read(2))[0]
                    f.seek(ln - 2, 1)
    except Exception:
        return None
    return None
