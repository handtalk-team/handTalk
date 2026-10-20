"""
시연 노트북처럼 인터넷이 없는 곳에서도 웹앱이 열리도록, 공개 라이브러리 파일을 client/vendor/ 에 받아 둔다.
(three.js r147 + GLTFLoader, MediaPipe Hands/카메라/그리기, Pretendard 글꼴)  — 한 번만 실행

    python client/tools/fetch_vendor.py
"""

from __future__ import annotations

import os
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor"))
CDN = "https://cdn.jsdelivr.net"
HANDS = "npm/@mediapipe/hands@0.4.1675469240"
FILES = {
    "three/three.min.js": f"{CDN}/npm/three@0.147.0/build/three.min.js",
    "three/GLTFLoader.js": f"{CDN}/npm/three@0.147.0/examples/js/loaders/GLTFLoader.js",
    "mediapipe/camera_utils.js": f"{CDN}/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js",
    "mediapipe/drawing_utils.js": f"{CDN}/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js",
    "fonts/PretendardVariable.woff2": f"{CDN}/npm/pretendard@1.3.9/dist/web/variable/woff2/PretendardVariable.woff2",
}
for f in ["hands.js", "hands.binarypb", "hand_landmark_full.tflite", "hand_landmark_lite.tflite",
          "hands_solution_packed_assets.data", "hands_solution_packed_assets_loader.js",
          "hands_solution_simd_wasm_bin.js", "hands_solution_simd_wasm_bin.wasm",
          "hands_solution_wasm_bin.js", "hands_solution_wasm_bin.wasm"]:
    FILES[f"mediapipe/hands/{f}"] = f"{CDN}/{HANDS}/{f}"

FONT_CSS = """@font-face {
  font-family: 'Pretendard Variable';
  font-weight: 45 920;
  font-style: normal;
  font-display: swap;
  src: url('PretendardVariable.woff2') format('woff2-variations');
}
"""


def main() -> None:
    total = 0
    for rel, url in FILES.items():
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            total += os.path.getsize(dst)
            continue
        with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
        total += os.path.getsize(dst)
        print(f"{rel}  {os.path.getsize(dst) / 1e6:.2f} MB")
    with open(os.path.join(ROOT, "fonts", "pretendard.css"), "w", encoding="utf-8") as f:
        f.write(FONT_CSS)
    print(f"done: {total / 1e6:.1f} MB in {ROOT}")


if __name__ == "__main__":
    main()
