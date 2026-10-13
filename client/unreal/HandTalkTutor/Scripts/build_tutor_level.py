"""
실시간 튜터 레벨 (/Game/HandTalk/Live/TutorLive) — Pixel Streaming 으로 웹 앱 선생님 칸에 보내는 화면.
Vivian(옷 입은 메타휴먼) + 조명 + 상반신 카메라 + AHandTalkAvatarDriver(서버 /ws/avatar 에 연결해 클립 재생).

에디터 없이 만들기:
    UnrealEditor.exe HandTalkTutor.uproject -ExecCmds="py <이 파일>"   (끝나면 에디터가 닫힘)
실행 (화면 없이 스트리밍):
    Scripts/run_live_tutor.bat
"""
import unreal

LEVEL = "/Game/HandTalk/Live/TutorLive"
MH_BP = "/Game/MetaHumans/Vivian/BP_Vivian"
PARTS = {p: f"/Game/HandTalk/SignsViv/{p}" for p in ("Body", "Torso", "Legs", "Feet")}

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)

if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
    try:
        les.load_level(LEVEL)
    except RuntimeError as e:
        unreal.log_warning(f"[handtalk] load_level: {e}")
    for a in eas.get_all_level_actors():
        eas.destroy_actor(a)
else:
    les.new_level(LEVEL)

plane = unreal.load_asset("/Engine/BasicShapes/Plane")
floor = eas.spawn_actor_from_object(plane, unreal.Vector(0, 0, 0))
floor.set_actor_scale3d(unreal.Vector(20, 20, 1))
back = eas.spawn_actor_from_object(plane, unreal.Vector(0, -300, 300))
back.set_actor_scale3d(unreal.Vector(20, 20, 1))
back.set_actor_rotation(unreal.Rotator(roll=90, pitch=0, yaw=0), False)

sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(roll=0, pitch=-40, yaw=-60))
sun.light_component.set_intensity(6.0)
sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 300))
sky.light_component.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
sky.light_component.set_editor_property("cubemap", unreal.load_asset("/Engine/MapTemplates/Sky/SunsetAmbientCubemap"))
sky.light_component.set_intensity(1.5)
eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0)).set_editor_property("unbound", True)

tutor = eas.spawn_actor_from_class(unreal.EditorAssetLibrary.load_blueprint_class(MH_BP), unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
tutor.set_actor_label("Tutor")

# 웹 앱 선생님 칸은 세로로 길어서 세로(9:16) 화면으로 렌더한다 (run_live_tutor.bat 의 ResX/ResY 와 같은 비율)
cam = eas.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(0, 230, 132), unreal.Rotator(roll=0, pitch=-2, yaw=-90))
cc = cam.get_cine_camera_component()
fb = cc.get_editor_property("filmback")
fb.sensor_width, fb.sensor_height = 13.5, 24.0
cc.set_editor_property("filmback", fb)
cc.set_editor_property("current_focal_length", 42.0)   # 세로 약 130cm: 머리부터 허리까지
pp = cc.get_editor_property("post_process_settings")
pp.set_editor_property("override_auto_exposure_bias", True)
pp.set_editor_property("auto_exposure_bias", 0.8)
cc.set_editor_property("post_process_settings", pp)
fs = cc.get_editor_property("focus_settings")
fs.focus_method = unreal.CameraFocusMethod.DISABLE
cc.set_editor_property("focus_settings", fs)
cam.set_actor_label("TutorCamera")

driver = eas.spawn_actor_from_class(unreal.HandTalkAvatarDriver, unreal.Vector(0, 0, 0))
driver.set_actor_label("AvatarDriver")
link = driver.get_editor_property("link")
link.set_editor_property("target_actor", tutor)
link.set_editor_property("view_camera", cam)
link.set_editor_property("part_clip_folders", PARTS)

les.save_current_level()
unreal.log("[handtalk] TutorLive level saved")
unreal.SystemLibrary.quit_editor()
