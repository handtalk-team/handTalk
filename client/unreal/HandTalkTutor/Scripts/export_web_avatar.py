"""
옷 입은 메타휴먼(Vivian)을 웹용 GLB 로 내보낸다 → client/assets/avatar/web/tutor.glb
브라우저 three.js 플레이어(client/src/components/web-avatar.js)가 이 파일에 서버 동작을 입힌다. 언리얼은 이때 한 번만 쓴다.

    UnrealEditor.exe HandTalkTutor.uproject -ExecCmds="py export_web_avatar.py --lod 1,QUIT_EDITOR" -unattended -nullrhi

--lod  내보낼 LOD (0 이 가장 정밀). 시연 노트북 내장 그래픽용으로 1~2 권장.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

import unreal

LEVEL = "/Game/HandTalk/Live/TutorLive"
HAIR_CARDS = {
    "hair": "/Game/MetaHumans/Vivian/MaleHair/Hair/Hair_M_SideSweptFringe_CardsMesh_Group0_LOD1",
    "eyebrows": "/Game/MetaHumans/Vivian/MaleHair_pfn/Hair/Eyebrows_M_Messy_CardsMesh_Group0_LOD1",
}
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "avatar", "web", "tutor.glb"))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lod", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tex", type=int, default=1024, help="굽는 텍스처 크기")
    ap.add_argument("--hair", action="store_true", help="머리카락·눈썹 카드 메시만 내보냄 (hair.glb, eyebrows.glb)")
    ap.add_argument("--source", action="store_true",
                    help="렌더 데이터 대신 원본 메시(MeshDescription)로 내보냄. 렌더 데이터는 본 영향 수 무제한/청킹 설정 때문에 "
                         "스킨 가중치가 깨져 나온다 (정점이 엉뚱한 본에 붙음).")
    args = ap.parse_args(argv)

    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(LEVEL)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    tutor = [a for a in eas.get_all_level_actors() if "vivian" in a.get_class().get_name().lower()
             or "vivian" in a.get_actor_label().lower()]
    if not tutor:
        raise RuntimeError("Vivian 액터를 찾지 못함")
    actor = tutor[0]
    unreal.log(f"[handtalk] export {actor.get_actor_label()} lod={args.lod}")

    opt = unreal.GLTFExportOptions()
    for name, value in [
        ("export_vertex_skin_weights", True),
        ("export_morph_targets", True),
        ("export_animation_sequences", False),
        ("export_level_sequences", False),
        ("default_level_of_detail", args.lod),
        ("bake_material_inputs", unreal.GLTFMaterialBakeMode.USE_MESH_DATA),
        ("default_material_bake_size", unreal.GLTFMaterialBakeSize(x=args.tex, y=args.tex, auto_detect=False)),
        ("texture_image_format", unreal.GLTFTextureImageFormat.JPEG),
        ("export_source_model", args.source),
        ("export_cameras", False),
        ("export_lights", False),
    ]:
        if value is None:
            continue
        try:
            opt.set_editor_property(name, value)
        except Exception as e:   # 엔진 버전에 따라 없는 옵션은 건너뜀
            unreal.log_warning(f"[handtalk] option {name}: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.hair:
        # 그룸(머리카락) 컴포넌트는 glTF 로 안 나가므로, 같은 머리 모양의 카드 메시(정적 메시)를 따로 내보낸다
        for name, path in HAIR_CARDS.items():
            mesh = unreal.load_asset(path)
            if not mesh:
                unreal.log_warning(f"[handtalk] 없음 {path}")
                continue
            dst = os.path.join(os.path.dirname(args.out), f"{name}.glb")
            res = unreal.GLTFExporter.export_to_gltf(mesh, dst, opt, set())
            unreal.log(f"[handtalk] hair {name} -> {dst} ({os.path.getsize(dst) if os.path.exists(dst) else 0} bytes)")
        return
    world = unreal.EditorLevelLibrary.get_editor_world()
    res = unreal.GLTFExporter.export_to_gltf(world, args.out, opt, {actor})
    ok, msgs = (res if isinstance(res, tuple) else (res, None))
    if msgs is not None:
        for kind in ("warnings", "errors"):
            for m in (msgs.get_editor_property(kind) or [])[:30]:
                unreal.log_warning(f"[handtalk] gltf {kind}: {m}")
    unreal.log(f"[handtalk] glTF export ok={ok} -> {args.out} ({os.path.getsize(args.out) if os.path.exists(args.out) else 0} bytes)")


if __name__ == "__main__":
    main(shlex.split(" ".join(sys.argv[1:]), posix=False) if len(sys.argv) > 1 else [])
