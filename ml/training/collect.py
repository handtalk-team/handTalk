"""
Data Collection Script
======================
Records fused sensor windows directly into ml/data/raw/{label}/.

Usage
-----
    python -m ml.training.collect --label 안녕하세요 --samples 300

Controls
--------
  SPACE  : record one sign attempt (hold during gesture)
  ENTER  : save the recording
  ESC    : quit

What gets saved
---------------
One (T, 77) float32 .npy file per accepted attempt.
T = number of frames captured during the SPACE press.
The file name is {timestamp}_{seq}.npy.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.services.recognition.glove import MockGloveSensor
from app.services.sync.fusion import SensorFusionModule
from app.models.schemas.sensor import SensorFrame

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def collect(label: str, target: int, use_mock_glove: bool = True) -> None:
    save_dir = Path(f"ml/data/raw/{label}")
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(save_dir.glob("*.npy")))
    print(f"[collect] label='{label}'  existing={existing}  target={target}")

    import asyncio
    glove_sensor = MockGloveSensor(hz=50)
    asyncio.get_event_loop().run_until_complete(glove_sensor.start())

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    saved = 0
    recording = False
    buffer: list = []
    fusion = SensorFusionModule()
    seq = 0

    print("Press SPACE to start/stop recording, Q to quit.")

    while saved < target:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        vis_data = None
        if result.multi_hand_landmarks:
            lms  = result.multi_hand_landmarks[0]
            wlms = result.multi_hand_world_landmarks[0] if result.multi_hand_world_landmarks else lms
            hand = result.multi_handedness[0].classification[0].label if result.multi_handedness else "Right"
            conf = result.multi_handedness[0].classification[0].score if result.multi_handedness else 0.9

            from app.models.schemas.sensor import VisionData, HandLandmark
            vis_data = VisionData(
                landmarks=[HandLandmark(x=l.x, y=l.y, z=l.z) for l in lms.landmark],
                world_landmarks=[HandLandmark(x=l.x, y=l.y, z=l.z) for l in wlms.landmark],
                confidence=float(conf),
                handedness=hand,
                fps=30.0,
            )
            mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if recording and vis_data:
            glove_data = asyncio.get_event_loop().run_until_complete(glove_sensor.read())
            sf = SensorFrame(
                timestamp=time.time(),
                sequence=seq,
                session_id="collect",
                camera=vis_data,
                glove=glove_data,
            )
            seq += 1
            ff = fusion.push_frame(sf)
            buffer.append(ff.fused_features.copy())

        # HUD
        h, w = frame.shape[:2]
        status = f"[REC {'●' if recording else '○'}]  saved={saved}/{target}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.putText(frame, f"Label: {label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        if buffer:
            cv2.putText(frame, f"frames: {len(buffer)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

        cv2.imshow("handTalk — Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                recording = True
                buffer = []
                fusion.reset()
                print("  → Recording started")
            else:
                recording = False
                if len(buffer) >= 10:
                    arr = np.array(buffer, dtype=np.float32)
                    fname = save_dir / f"{int(time.time())}_{uuid.uuid4().hex[:6]}.npy"
                    np.save(str(fname), arr)
                    saved += 1
                    print(f"  → Saved ({len(buffer)} frames) → {fname}   [{saved}/{target}]")
                else:
                    print(f"  → Too short ({len(buffer)} frames), discarded")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    asyncio.get_event_loop().run_until_complete(glove_sensor.stop())
    print(f"\nCollection done. Total saved: {saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   required=True, help="Sign label in Korean, e.g. 안녕하세요")
    parser.add_argument("--samples", type=int, default=300, help="Target number of samples")
    parser.add_argument("--mock",    action="store_true", default=True,
                        help="Use mock glove (default True until real glove connected)")
    args = parser.parse_args()
    collect(args.label, args.samples, args.mock)
