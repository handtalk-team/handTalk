"""
AI Hub 모션 zip(단어 이름.motion.json) → 클립 이름으로 바꿔 모션 폴더에 넣는다. (에디터 밖, 일반 파이썬)

    python import_motion_zip.py <zip> <out_dir>

단어 → 클립 이름은 server/app/services/tutor/curriculum.py 의 GLOSS_TO_CLIP 과 같다.
이어서 build_vivian_signs.py / build_metahuman_signs.py 로 언리얼 클립을 만든다.
"""
import json, os, re, sys, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CUR = open(os.path.join(ROOT, "server", "app", "services", "tutor", "curriculum.py"), encoding="utf-8").read()
block = CUR[CUR.index("GLOSS_TO_CLIP"):]
block = block[:block.index("}") + 1]
MAP = dict(re.findall(r'"([^"]+)":\s*"(KSL_[^"]+)"', block))
# 노트북이 요청 단어 이름으로 저장한 대체 라벨 파일 (AI Hub 에 없는 단어 → 가장 가까운 라벨)
ALIAS = {"미안하다": "죄송", "기쁘다": "즐겁다"}

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
done = []
with zipfile.ZipFile(src) as z:
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        word = os.path.basename(name).split(".")[0]
        word = ALIAS.get(word, word)
        clip = MAP.get(word)
        if not clip:
            print("skip (no clip name):", name)
            continue
        m = json.loads(z.read(name).decode("utf-8"))
        m["clip"] = clip
        m.setdefault("label", word)
        with open(os.path.join(out, f"{clip}.motion.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False)
        done.append(f"{word}→{clip}({len(m['frames'])}f)")
print(len(done), "imported:", ", ".join(done))
