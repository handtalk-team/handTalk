# handTalk Unreal 클라이언트 (MetaHuman 튜터)

UE 5.7 C++ 프로젝트 `HandTalkTutor`. 서버의 `/ws/avatar` 에 붙어 튜터 응답(`llm_response.avatar_commands`)을 받고,
clip 이름과 같은 AnimSequence 를 `/Game/HandTalk/Signs/` 에서 찾아 메타휴먼 Body 에 재생한다.

메타휴먼 에셋과 `Content/` 는 저장소에 넣지 않는다 (`.gitignore`). 코드, 설정, 스크립트만 올린다.

## 구성

| 파일 | 역할 |
|------|------|
| `Source/HandTalkTutor/HandTalkAvatarLink.*` | 액터 컴포넌트. WebSocket 연결, 자동 재연결, 클립 큐 재생. 블루프린트 이벤트 `OnTutorReply`, `OnRecognition`, `OnClipStarted` |
| `Source/HandTalkTutor/HandTalkSkeletonLibrary.*` | 에디터 파이썬이 스켈레톤 부모 관계를 읽는 함수 |
| `Scripts/sign_retarget.py` | 수어자 좌표 키포인트 → 본 회전 (언리얼 없이 테스트 가능한 순수 파이썬) |
| `Scripts/build_sign_anim.py` | 모션 JSON → AnimSequence 생성 (에디터 파이썬) |

## AI Hub 키포인트로 수어 클립 만들기

```powershell
# 1) 클립 폴더 → 모션 JSON (저장소 루트에서)
python -m ml.avatar.aihub_motion --clip <클립 폴더> --morpheme <클립>_morpheme.json --clip_name KSL_hello --out C:\handtalk\motions

# 2) 모션 JSON → AnimSequence (에디터 없이)
UnrealEditor-Cmd.exe client\unreal\HandTalkTutor\HandTalkTutor.uproject -run=pythonscript -script="client/unreal/HandTalkTutor/Scripts/build_sign_anim.py --motion C:/handtalk/motions --skeleton <메타휴먼 Body 스켈레톤 경로>"
```

`--skeleton` 을 비우면 이름에 `metahuman` 이 들어간 스켈레톤을 찾는다. 팔, 손, 손가락만 움직이고
척추와 머리는 참조 포즈를 유지한다.

## 레벨에 붙이기

1. 메타휴먼 블루프린트를 레벨에 놓는다.
2. 컴포넌트 추가에서 `HandTalk Avatar Link` 를 붙인다. `Target Mesh` 를 비우면 이름에 Body 가 들어간 메시를 쓴다.
3. 서버를 띄우고 (`uvicorn app.main:app --port 8000`) Play 하면 `ws://127.0.0.1:8000/ws/avatar` 에 연결된다.
4. 서버 없이 시험하려면 블루프린트에서 `Play Clip By Name` 을 호출한다.
