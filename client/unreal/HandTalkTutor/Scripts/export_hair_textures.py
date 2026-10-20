"""머리카락·눈썹 카드 아틀라스 텍스처를 PNG 로 내보낸다 (웹 아바타 머리카락 투명도용) → client/assets/avatar/web/tex/"""
import os
import unreal

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "avatar", "web", "tex"))
TEX = [
    "/Game/MetaHumans/Common/MaleHair/Textures/Hair_M_SideSweptFringe_CardsAtlas_Attribute",
    "/Game/MetaHumans/Common/MaleHair_pfn/Textures/Eyebrows_M_Messy_CardsAtlas_Attribute",
]
os.makedirs(OUT, exist_ok=True)
for path in TEX:
    tex = unreal.load_asset(path)
    task = unreal.AssetExportTask()
    task.object = tex
    task.filename = os.path.join(OUT, path.rsplit("/", 1)[1] + ".tga")
    task.automated = True
    task.replace_identical = True
    task.prompt = False
    ok = unreal.Exporter.run_asset_export_task(task)
    unreal.log(f"[handtalk] tex {path} ok={ok} -> {task.filename}")
