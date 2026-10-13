"""메타휴먼(MHC_Tutor) Body 스켈레톤용 수어 클립을 /Game/HandTalk/SignsMH 에 만든다. 에디터 Cmd: py "<이 파일>" """
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sign_anim as b

b.main(["--motion", "C:/handtalk/motions",
        "--skeleton", "/Game/MetaHumans/Common/Female/Medium/NormalWeight/Body/metahuman_base_skel",
        "--folder", "/Game/HandTalk/SignsMH"])
