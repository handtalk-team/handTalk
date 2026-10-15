"""
메타휴먼 Body 메시의 참조 포즈(컴포넌트 공간 위치/회전, 부모)를 JSON 으로 내보낸다.
서버가 이 파일로 키포인트 → 본 회전을 직접 계산해 언리얼에 실시간으로 보낸다 (server/app/services/avatar/motion_synth.py).

    UnrealEditor.exe HandTalkTutor.uproject -ExecCmds="py export_ref_pose.py --mesh /Game/MetaHumans/Vivian/Body/f_tal_nrw_body --out C:/handtalk/handTalk/client/assets/avatar/ref_pose.json,QUIT_EDITOR"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sign_anim import read_mesh_ref_pose  # noqa: E402


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="/Game/MetaHumans/Vivian/Body/f_tal_nrw_body")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ref, _local = read_mesh_ref_pose(args.mesh)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mesh": args.mesh, "pos": ref.pos, "rot": ref.rot, "parent": ref.parent}, f)
    unreal.log(f"[handtalk] ref pose {len(ref.pos)} bones -> {args.out}")


if __name__ == "__main__":
    main(shlex.split(" ".join(sys.argv[1:]), posix=False) if len(sys.argv) > 1 else [])
