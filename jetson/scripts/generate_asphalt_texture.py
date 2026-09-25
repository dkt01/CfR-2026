#!/usr/bin/env python3
"""Create deterministic, world-scale asphalt parking lot textures for Gazebo."""

import math
from pathlib import Path
import random
import struct
import zlib


PIXELS_PER_METER = 32
MATERIALS = Path(__file__).parents[1] / "cfr_arduino_bridge/materials"
COURSES = {"speed": (60, 45), "obstacle": (30, 24)}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return (
        struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))
    )


def make_texture(name: str, width_m: int, height_m: int, seed: int) -> None:
    width = width_m * PIXELS_PER_METER
    height = height_m * PIXELS_PER_METER
    rng = random.Random(seed)
    patches = [
        [rng.randrange(-9, 10) for _ in range(width // 16 + 1)]
        for _ in range(height // 16 + 1)
    ]
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter type 0
        py, fy = divmod(y, 16)
        y_m = y / PIXELS_PER_METER
        # A 6 m driving aisle has 4.8 m deep parking bays on both sides.
        # Successive aisles run in opposite directions, so their angled
        # spaces reverse direction while each aisle has bays on both sides.
        aisle_number = math.floor(y_m / 15.6)
        pair_y = y_m - aisle_number * 15.6
        top = pair_y < 4.8
        bottom = pair_y >= 10.8
        in_stall = top or bottom
        curb = pair_y < 0.10 or pair_y > 15.5
        if top:
            offset_m = (4.8 - pair_y) / math.sqrt(3)
        elif bottom:
            offset_m = (pair_y - 10.8) / math.sqrt(3)
        else:
            offset_m = 0
        if aisle_number % 2:
            offset_m = -offset_m
        for x in range(width):
            px, fx = divmod(x, 16)
            a = patches[py][px] * (16 - fx) + patches[py][px + 1] * fx
            b = patches[py + 1][px] * (16 - fx) + patches[py + 1][px + 1] * fx
            patch = (a * (16 - fy) + b * fy) // 256
            grain = rng.randrange(-11, 12)
            aggregate = rng.randrange(12, 25) if rng.randrange(80) == 0 else 0
            gray = max(0, min(255, 78 + patch + grain + aggregate))
            # Each stall is 3 m wide along the row; dividers meet the aisle
            # at 60 degrees on both sides.
            divider = in_stall and ((x / PIXELS_PER_METER - offset_m) % 3 < 0.10)
            if (curb or divider) and rng.randrange(30) != 0:
                wear = grain + patch
                rows.extend((210 + wear, 169 + wear, 35 + wear))
            else:
                rows.extend((gray, gray, gray))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    )
    png.extend(png_chunk(b"IDAT", zlib.compress(rows, level=9)))
    png.extend(png_chunk(b"IEND", b""))
    output = MATERIALS / f"asphalt_{name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
    print(f"wrote {output}")


def main() -> None:
    for index, (name, (width, height)) in enumerate(COURSES.items()):
        make_texture(name, width, height, 2026 + index)


if __name__ == "__main__":
    main()
