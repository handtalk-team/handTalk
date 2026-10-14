"""
handTalk 모션 JSON → 메타휴먼 AnimSequence (언리얼 에디터 파이썬)
================================================================

ml/avatar/aihub_motion.py 가 만든 *.motion.json 을 읽어 메타휴먼 Body 스켈레톤용
AnimSequence 를 /Game/HandTalk/Signs/{clip} 에 만든다. 이름은 서버가 보내는 clip 이름과 같다.

에디터에서 실행 (Output Log 의 Cmd 를 Python 으로 바꾸고):
    py "C:/handtalk/handTalk/client/unreal/HandTalkTutor/Scripts/build_sign_anim.py" --motion "C:/handtalk/motions" --skeleton "/Game/MetaHumans/Common/Female/Medium/NormalWeight/Body/metahuman_base_skel"

명령줄 (에디터 창 없이):
    UnrealEditor-Cmd.exe HandTalkTutor.uproject -run=pythonscript -script="build_sign_anim.py --motion ... --skeleton ..."

--motion 은 파일 하나 또는 *.motion.json 이 든 폴더.
--skeleton 을 비우면 프로젝트에서 이름에 metahuman 이 들어간 스켈레톤을 찾는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sign_retarget as sr  # noqa: E402

DEFAULT_FOLDER = "/Game/HandTalk/Signs"


def find_skeleton(path: str | None) -> unreal.Skeleton:
    if path:
        sk = unreal.load_asset(path)
        if not isinstance(sk, unreal.Skeleton):
            raise RuntimeError(f"스켈레톤이 아님: {path}")
        return sk
    reg = unreal.AssetRegistryHelpers.get_asset_registry()
    flt = unreal.ARFilter(class_paths=[unreal.TopLevelAssetPath("/Script/Engine", "Skeleton")],
                          package_paths=["/Game"], recursive_paths=True)
    cands = [a for a in reg.get_assets(flt)]
    for a in cands:
        if "metahuman" in str(a.asset_name).lower() and "face" not in str(a.asset_name).lower():
            return a.get_asset()
    for a in cands:   # 마네킹으로 테스트할 때
        if "manny" in str(a.asset_name).lower() or "mannequin" in str(a.asset_name).lower():
            return a.get_asset()
    raise RuntimeError("스켈레톤을 찾지 못함. --skeleton 으로 지정하세요.")


def _vec(v) -> sr.Vec:
    return (v.x, v.y, v.z)


def _quat(q) -> sr.Quat:
    return (q.x, q.y, q.z, q.w)


def read_ref_pose(skeleton: unreal.Skeleton) -> tuple[sr.RefPose, dict]:
    pose = unreal.AnimPoseExtensions.get_reference_pose(skeleton)
    names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose)]
    world_pos, world_rot, local_t, parent = {}, {}, {}, {}
    W = unreal.AnimPoseSpaces.WORLD
    L = unreal.AnimPoseSpaces.LOCAL
    for n in names:
        tw = unreal.AnimPoseExtensions.get_ref_bone_pose(pose, n, W)
        tl = unreal.AnimPoseExtensions.get_ref_bone_pose(pose, n, L)
        world_pos[n] = _vec(tw.translation)
        world_rot[n] = _quat(tw.rotation)
        local_t[n] = tl
    # AnimPose 는 부모 이름을 노출하지 않아 프로젝트 C++ 함수로 읽는다
    bones, parents = unreal.HandTalkSkeletonLibrary.get_bone_hierarchy(skeleton)
    for b, p in zip(bones, parents):
        parent[str(b)] = None if str(p) == "None" else str(p)
    return sr.RefPose(world_pos, world_rot, parent), local_t


def read_mesh_ref_pose(mesh_path: str) -> tuple[sr.RefPose, dict]:
    """스켈레톤 대신 특정 메시의 참조 포즈 (같은 스켈레톤을 쓰는 다른 체형용, 예: 키 큰 Vivian)"""
    mod = unreal.SkeletonModifier()
    if not mod.set_skeletal_mesh(unreal.load_asset(mesh_path)):
        raise RuntimeError(f"메시를 읽지 못함: {mesh_path}")
    world_pos, world_rot, local_t, parent = {}, {}, {}, {}
    for b in mod.get_all_bone_names():
        n = str(b)
        tw = mod.get_bone_transform(b, True)
        world_pos[n] = _vec(tw.translation)
        world_rot[n] = _quat(tw.rotation)
        local_t[n] = mod.get_bone_transform(b, False)
        p = str(mod.get_parent_name(b))
        parent[n] = None if p == "None" else p
    return sr.RefPose(world_pos, world_rot, parent), local_t


def build_one(motion_path: str, skeleton: unreal.Skeleton, ref: sr.RefPose, local_t: dict,
              folder: str) -> str:
    with open(motion_path, encoding="utf-8") as f:
        motion = json.load(f)
    clip = motion.get("clip") or os.path.basename(motion_path).split(".")[0]
    rt = sr.Retargeter(ref)
    if rt.missing:
        unreal.log_warning(f"[handtalk] 스켈레톤에 없는 본 (건너뜀): {rt.missing}")
    tracks = rt.solve_motion(motion)
    n = len(motion["frames"])
    fps = int(round(motion.get("fps", 30)))
    if abs(fps - 30) <= 2:   # AI Hub 영상은 30fps (프레임 수/길이로 추정하면 29.x 가 나올 수 있음)
        fps = 30

    asset_path = f"{folder}/{clip}"
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        unreal.EditorAssetLibrary.delete_asset(asset_path)
    factory = unreal.AnimSequenceFactory()
    factory.set_editor_property("target_skeleton", skeleton)
    anim = unreal.AssetToolsHelpers.get_asset_tools().create_asset(clip, folder, unreal.AnimSequence, factory)

    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket(unreal.Text(f"handtalk {clip}"), False)
    ctrl.set_frame_rate(unreal.FrameRate(fps, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(max(n - 1, 1)), False)
    for bone, quats in tracks.items():
        t = local_t[bone]
        pos = [t.translation] * n
        scl = [t.scale3d] * n
        rot = [unreal.Quat(q[0], q[1], q[2], q[3]) for q in quats]
        ctrl.add_bone_curve(bone, False)
        ctrl.set_bone_track_keys(bone, pos, rot, scl, False)
    ctrl.close_bracket(False)

    unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False)
    unreal.log(f"[handtalk] {asset_path}  frames={n} fps={fps} bones={len(tracks)}")
    return asset_path


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--skeleton", default=None)
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--mesh", default=None, help="참조 포즈를 이 메시에서 읽음 (체형이 다른 메타휴먼)")
    ap.add_argument("--skip-existing", action="store_true", help="이미 있는 클립은 건너뜀 (수천 개 일괄 생성용)")
    args = ap.parse_args(argv)

    files = (sorted(glob.glob(os.path.join(args.motion, "*.motion.json")))
             if os.path.isdir(args.motion) else [args.motion])
    if not files:
        raise RuntimeError(f"모션 파일 없음: {args.motion}")

    skeleton = find_skeleton(args.skeleton)
    unreal.log(f"[handtalk] skeleton={skeleton.get_path_name()}")
    ref, local_t = read_mesh_ref_pose(args.mesh) if args.mesh else read_ref_pose(skeleton)
    made = skipped = failed = 0
    for i, f in enumerate(files, 1):
        clip = os.path.basename(f).split(".")[0]
        if args.skip_existing and unreal.EditorAssetLibrary.does_asset_exist(f"{args.folder}/{clip}"):
            skipped += 1
            continue
        try:
            build_one(f, skeleton, ref, local_t, args.folder)
            made += 1
        except Exception as e:   # 한 클립이 실패해도 나머지는 계속
            failed += 1
            unreal.log_warning(f"[handtalk] {clip} 실패: {e}")
        if i % 200 == 0:
            unreal.log(f"[handtalk] progress {i}/{len(files)} made={made} skipped={skipped} failed={failed}")
    unreal.log(f"[handtalk] done {args.folder}: made={made} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main(shlex.split(" ".join(sys.argv[1:]), posix=False) if len(sys.argv) > 1 else [])
