"""Vivian(키 큰 체형, 셔츠/청바지/운동화)용 수어 클립. 몸과 옷 메시가 체형과 스켈레톤이 달라
부위마다 그 메시의 참조 포즈로 같은 클립을 만든다 → /Game/HandTalk/SignsViv/{Body,Torso,Legs,Feet}/KSL_*
에디터 Cmd: py "<이 파일>" """
import os, sys
import unreal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sign_anim as b

PARTS = {
    "Body": "/Game/MetaHumans/Vivian/Body/f_tal_nrw_body",
    "Torso": "/Game/MetaHumans/Common/Female/Tall/NormalWeight/Tops/Shirt/f_tal_nrw_top_shirt_nrm_Cinematic",
    "Legs": "/Game/MetaHumans/Common/Female/Tall/NormalWeight/Bottoms/Jeans/f_tal_nrw_btm_jeans_slm_Cinematic",
    "Feet": "/Game/MetaHumans/Common/Female/Tall/NormalWeight/Shoes/CasualSneakers/f_tal_nrw_shs_casualsneakers_Cinematic",
}
# 수어 동작은 팔과 손만 움직이므로 하의/신발 클립은 처음 만든 단어들에만 있다.
# 수천 단어를 일괄로 만들 때는 몸과 상의만 만들고 이미 있는 클립은 건너뛴다.
BULK = os.environ.get("HANDTALK_BULK", "1") == "1"
for part, mesh_path in PARTS.items():
    if BULK and part in ("Legs", "Feet"):
        continue
    sk = unreal.load_asset(mesh_path).get_editor_property("skeleton")
    b.main(["--motion", "C:/handtalk/motions", "--skeleton", sk.get_path_name().split(".")[0],
            "--mesh", mesh_path, "--folder", f"/Game/HandTalk/SignsViv/{part}"] + (["--skip-existing"] if BULK else []))
