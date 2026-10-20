"""
언리얼에서 내보낸 GLB 를 three.js 가 그릴 수 있게 손본다 (언리얼 없이 일반 파이썬으로 실행).

메타휴먼은 정점 하나에 뼈가 8~12개까지 붙는데(JOINTS_0..2), three.js 는 앞의 4개(JOINTS_0/WEIGHTS_0)만 쓴다.
그대로 두면 나머지 가중치가 사라져 정점이 원점으로 끌려간다. 정점마다 가장 큰 가중치 4개만 남기고 합이 1이 되게
다시 나눈 뒤 JOINTS_1 이후는 지운다.

    python fix_web_avatar.py client/assets/avatar/web/tutor.glb
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

CT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path):
    b = open(path, "rb").read()
    jl = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack("<I", b[off:off + 4])[0]
    return j, bytearray(b[off + 8:off + 8 + bl])


def write_glb(path, j, binbuf):
    while len(binbuf) % 4:
        binbuf += b"\0"
    j["buffers"][0]["byteLength"] = len(binbuf)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binbuf))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf)
    open(path, "wb").write(out)


def get(j, buf, idx):
    a = j["accessors"][idx]
    bv = j["bufferViews"][a["bufferView"]]
    dt = CT[a["componentType"]]
    n = NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    item = np.dtype(dt).itemsize * n
    if stride and stride != item:
        raw = np.frombuffer(bytes(buf[start:start + stride * a["count"]]), np.uint8).reshape(a["count"], stride)[:, :item]
        return np.frombuffer(raw.tobytes(), dt).reshape(a["count"], n)
    return np.frombuffer(bytes(buf[start:start + item * a["count"]]), dt).reshape(a["count"], n).copy()


def add(j, buf, arr, ctype, atype, normalized=False):
    data = np.ascontiguousarray(arr).tobytes()
    while len(buf) % 4:
        buf += b"\0"
    off = len(buf)
    buf += data
    j["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": 34962})
    acc = {"bufferView": len(j["bufferViews"]) - 1, "componentType": ctype, "count": int(arr.shape[0]), "type": atype}
    if normalized:
        acc["normalized"] = True
    j["accessors"].append(acc)
    return len(j["accessors"]) - 1


def main(path: str) -> None:
    j, buf = read_glb(path)
    fixed = 0
    for mesh in j["meshes"]:
        for prim in mesh["primitives"]:
            at = prim["attributes"]
            sets = sorted(k[-1] for k in at if k.startswith("JOINTS_"))
            if not sets:
                continue
            J = np.concatenate([get(j, buf, at[f"JOINTS_{s}"]).astype(np.int64) for s in sets], 1)
            W = np.concatenate([get(j, buf, at[f"WEIGHTS_{s}"]).astype(np.float64) for s in sets], 1)
            wacc = j["accessors"][at["WEIGHTS_0"]]
            if wacc.get("normalized"):
                W /= 255.0 if wacc["componentType"] == 5121 else 65535.0
            top = np.argsort(-W, 1)[:, :4]
            J4 = np.take_along_axis(J, top, 1)
            W4 = np.take_along_axis(W, top, 1)
            s = W4.sum(1, keepdims=True)
            W4 = np.where(s > 0, W4 / np.maximum(s, 1e-9), np.array([[1.0, 0, 0, 0]]))
            for k in [k for k in list(at) if k.startswith(("JOINTS_", "WEIGHTS_"))]:
                del at[k]
            at["JOINTS_0"] = add(j, buf, J4.astype(np.uint16), 5123, "VEC4")
            at["WEIGHTS_0"] = add(j, buf, W4.astype(np.float32), 5126, "VEC4")
            fixed += 1
    write_glb(path, j, buf)
    print(f"fixed {fixed} primitives -> {path}")


if __name__ == "__main__":
    main(sys.argv[1])
