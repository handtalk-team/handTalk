"""
수어 클립 시연 영상 렌더 (언리얼 에디터 파이썬)
================================================

/Game/HandTalk/Signs 의 클립들을 마네킹(또는 지정한 메시)이 차례로 재생하는 레벨 시퀀스를 만들고
Movie Render Queue 로 PNG 프레임을 렌더한다. 프레임 → 영상 변환은 에디터 밖에서 한다.

에디터 Cmd 창에서:
    py "C:/handtalk/handTalk/client/unreal/HandTalkTutor/Scripts/render_sign_demo.py"
"""

import sys
import unreal

VIVIAN = "--vivian" in sys.argv          # 옷 입은 기존 메타휴먼 BP_Vivian (키 큰 체형)
METAHUMAN = "--metahuman" in sys.argv or VIVIAN   # 마네킹 대신 메타휴먼으로 렌더
CLIPS = ["KSL_hospital", "KSL_hurt", "KSL_doctor", "KSL_okay", "KSL_thank_you"]
MESH = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple"
MH_BP = "/Game/MetaHumans/Vivian/BP_Vivian" if VIVIAN else "/Game/MetaHumans/MHC_Tutor/BP_MHC_Tutor"
CLIP_FOLDER = "/Game/HandTalk/SignsViv/Body" if VIVIAN else "/Game/HandTalk/SignsMH" if METAHUMAN else "/Game/HandTalk/Signs"
OUT_DIR = "C:/handtalk/render/frames_viv" if VIVIAN else "C:/handtalk/render/frames_mh" if METAHUMAN else "C:/handtalk/render/frames"
LEVEL = "/Game/HandTalk/Demo/SignDemo"
SEQ_PATH, SEQ_NAME = "/Game/HandTalk/Demo", "LS_SignDemoViv" if VIVIAN else "LS_SignDemoMH" if METAHUMAN else "LS_SignDemo"
FPS = 30
GAP = 0 if METAHUMAN else 12   # 클립 사이 쉬는 프레임 (비어 있으면 A포즈가 보여서 메타휴먼은 붙인다)
LEAD = 0 if METAHUMAN else 15  # 시작 전 대기 프레임
EXPOSURE = 0.5 if METAHUMAN else -1.0   # 자동 노출 보정 (흰 배경이라 어둡게)

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)

# ── 레벨: 바닥, 조명, 배경 ──
if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
    try:
        les.load_level(LEVEL)
    except RuntimeError as e:   # 메타휴먼 에셋 경고가 예외로 올라와도 레벨은 열린다
        unreal.log_warning(f"[handtalk] load_level: {e}")
    for a in eas.get_all_level_actors():
        eas.destroy_actor(a)
else:
    les.new_level(LEVEL)

floor = eas.spawn_actor_from_object(unreal.load_asset("/Engine/BasicShapes/Plane"), unreal.Vector(0, 0, 0))
floor.set_actor_scale3d(unreal.Vector(20, 20, 1))
backdrop = eas.spawn_actor_from_object(unreal.load_asset("/Engine/BasicShapes/Plane"), unreal.Vector(0, -300, 300),
                                       unreal.Rotator(0, 90, 0))
backdrop.set_actor_scale3d(unreal.Vector(20, 20, 1))
backdrop.set_actor_rotation(unreal.Rotator(roll=90, pitch=0, yaw=0), False)

sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(roll=0, pitch=-40, yaw=-60))
sun.light_component.set_intensity(6.0)
sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 300))
sky.light_component.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
sky.light_component.set_editor_property("cubemap", unreal.load_asset("/Engine/MapTemplates/Sky/SunsetAmbientCubemap"))
sky.light_component.set_intensity(1.5)
sky.light_component.recapture_sky()

# ── 캐릭터: 카메라(+Y 쪽)를 보게 ──
if METAHUMAN:
    char = eas.spawn_actor_from_class(unreal.EditorAssetLibrary.load_blueprint_class(MH_BP), unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    meshes = list(char.get_components_by_class(unreal.SkeletalMeshComponent))
    unreal.log(f"[handtalk] {MH_BP} components: {[c.get_name() for c in meshes]}")
    for c in meshes:   # 디버그: --hide=Torso 처럼 부위 숨기기
        if any(a == f"--hide={c.get_name()}" for a in sys.argv):
            c.set_editor_property("hidden_in_game", True)
    meshes = list(char.get_components_by_class(unreal.SkeletalMeshComponent))   # 속성 변경 시 BP 가 컴포넌트를 다시 만든다
    body = [m for m in meshes if m.get_name() == "Body"][0]
else:
    char = eas.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    char.skeletal_mesh_component.set_skinned_asset_and_update(unreal.load_asset(MESH))
    body = None
char.set_actor_label("SignTutor")
if VIVIAN:
    # 옷 메시(Torso, Legs, Feet)가 Body 를 따라가도록 Play 시점에 Leader Pose 를 거는 도우미
    lp = eas.spawn_actor_from_class(unreal.HandTalkLeaderPose, unreal.Vector(0, 0, 0))
    lp.set_editor_property("target", char)
    lp.set_editor_property("use_leader_pose", False)
    lp.set_editor_property("disable_post_process_on_leader", True)

# 마네킹은 +Y 를 앞으로 본다. 카메라를 +Y 쪽 가슴 높이에 둔다.
# 메타휴먼은 상반신(손 모양)이 잘 보이게 더 가까이
cam_pos, focal = (unreal.Vector(0, 210, 158), 30.0) if VIVIAN else (unreal.Vector(0, 200, 146), 30.0) if METAHUMAN else (unreal.Vector(0, 250, 128), 24.0)
cam = eas.spawn_actor_from_class(unreal.CineCameraActor, cam_pos, unreal.Rotator(roll=0, pitch=-4, yaw=-90))
cam.get_cine_camera_component().set_editor_property("current_focal_length", focal)
pp = cam.get_cine_camera_component().get_editor_property("post_process_settings")
pp.set_editor_property("override_auto_exposure_bias", True)
pp.set_editor_property("auto_exposure_bias", EXPOSURE)
cam.get_cine_camera_component().set_editor_property("post_process_settings", pp)
fs = cam.get_cine_camera_component().get_editor_property("focus_settings")
fs.focus_method = unreal.CameraFocusMethod.DISABLE
cam.get_cine_camera_component().set_editor_property("focus_settings", fs)

# ── 시퀀스 ──
full = f"{SEQ_PATH}/{SEQ_NAME}"
if unreal.EditorAssetLibrary.does_asset_exist(full):
    unreal.EditorAssetLibrary.delete_asset(full)
seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(SEQ_NAME, SEQ_PATH, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
seq.set_display_rate(unreal.FrameRate(FPS, 1))
seq.set_tick_resolution(unreal.FrameRate(FPS, 1))

bind = seq.add_possessable(char)
part_tracks = []
if VIVIAN:
    # 옷은 스켈레톤이 달라 부위별로 만든 같은 클립을 각 컴포넌트에 재생 (build_vivian_signs.py)
    for comp in meshes:
        if comp.get_name() in ("Torso", "Legs", "Feet"):
            part_tracks.append((seq.add_possessable(comp).add_track(unreal.MovieSceneSkeletalAnimationTrack),
                                f"/Game/HandTalk/SignsViv/{comp.get_name()}"))
if body is not None:
    bind = seq.add_possessable(body)   # 메타휴먼은 Body 컴포넌트에 애니메이션 트랙
track = bind.add_track(unreal.MovieSceneSkeletalAnimationTrack)
t = LEAD
marks = []
for name in CLIPS:
    anim = unreal.load_asset(f"{CLIP_FOLDER}/{name}")
    n = int(round(anim.get_play_length() * FPS)) + 1
    sec = track.add_section()
    sec.set_range(t, t + n)
    p = sec.get_editor_property("params")
    p.set_editor_property("animation", anim)
    sec.set_editor_property("params", p)
    for ptrack, pfolder in part_tracks:
        ps = ptrack.add_section()
        ps.set_range(t, t + n)
        pp_ = ps.get_editor_property("params")
        pp_.set_editor_property("animation", unreal.load_asset(f"{pfolder}/{name}"))
        ps.set_editor_property("params", pp_)
    marks.append((name, t, t + n))
    t += n + GAP
end = t
seq.set_playback_start(0)
seq.set_playback_end(end)

cam_bind = seq.add_possessable(cam)
cut = seq.add_master_track(unreal.MovieSceneCameraCutTrack) if hasattr(seq, "add_master_track") else seq.add_track(unreal.MovieSceneCameraCutTrack)
cs = cut.add_section()
cs.set_range(0, end)
cs.set_camera_binding_id(seq.get_binding_id(cam_bind) if hasattr(seq, "get_binding_id") else unreal.MovieSceneSequenceExtensions.get_portable_binding_id(seq, seq, cam_bind))

unreal.EditorAssetLibrary.save_asset(full)
les.save_current_level()

# 클립 구간 기록 (자막 넣을 때 사용)
import json, os
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(os.path.dirname(OUT_DIR), "marks_viv.json" if VIVIAN else "marks_mh.json" if METAHUMAN else "marks.json"), "w") as f:
    json.dump({"fps": FPS, "end": end, "marks": marks}, f)

unreal.log(f"[handtalk] scene ready: {full}  frames={end}")
if unreal.SystemLibrary.is_unattended() or not hasattr(unreal, "MoviePipelineQueueSubsystem"):
    # 명령줄(commandlet)에서는 장면만 만들고, 렌더는 -game -LevelSequence 로 따로 돌린다
    raise SystemExit(0)

# ── Movie Render Queue ──
mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
queue = mrq.get_queue()
for j in list(queue.get_jobs()):
    queue.delete_job(j)
job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
job.sequence = unreal.SoftObjectPath(full)
job.map = unreal.SoftObjectPath(LEVEL)
cfg = job.get_configuration()
out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
out.output_directory = unreal.DirectoryPath(OUT_DIR)
out.output_resolution = unreal.IntPoint(1280, 720)
out.file_name_format = "frame.{frame_number}"
out.use_custom_frame_rate = True
out.output_frame_rate = unreal.FrameRate(FPS, 1)
out.override_existing_output = True
cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
aa.spatial_sample_count = 1
aa.temporal_sample_count = 1

executor = mrq.render_queue_with_executor(unreal.MoviePipelinePIEExecutor)
unreal.log(f"[handtalk] render started: {end} frames → {OUT_DIR}")
