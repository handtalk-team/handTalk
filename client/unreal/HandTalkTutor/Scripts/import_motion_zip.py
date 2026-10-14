"""
AI Hub 모션 zip({단어}.motion.json 묶음) → 클립 이름으로 바꿔 모션 폴더에 넣고 단어 사전(lexicon.json)을 만든다.
(에디터 밖, 일반 파이썬)

    python import_motion_zip.py <zip> <out_dir> [--lexicon <lexicon.json 경로>]

클립 이름
  - server/app/services/tutor/curriculum.py 의 GLOSS_TO_CLIP 에 있는 단어는 그 이름 (KSL_hospital 등)
  - 나머지는 AI Hub 단어 ID 로 KSL_W1496 처럼 (모션의 source 에서 읽음, 없으면 순번)

lexicon.json 은 [{"gloss": 단어, "clip": 클립}] — 서버가 튜터 답의 글로스를 아바타 클립으로 바꿀 때 쓴다.
AI Hub 데이터에서 나온 목록이라 저장소에는 올리지 않는다 (.gitignore).
이어서 build_vivian_signs.py 로 언리얼 클립을 만든다.
"""
import argparse, json, os, re, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CUR = open(os.path.join(ROOT, "server", "app", "services", "tutor", "curriculum.py"), encoding="utf-8").read()
block = CUR[CUR.index("GLOSS_TO_CLIP"):]
block = block[:block.index("}") + 1]
MAP = dict(re.findall(r'"([^"]+)":\s*"(KSL_[^"]+)"', block))
# 노트북이 요청 단어 이름으로 저장한 대체 라벨 파일 (AI Hub 에 없는 단어 → 가장 가까운 라벨)
ALIAS = {"미안하다": "죄송", "기쁘다": "즐겁다", "감사합니다": "감사"}

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("out")
ap.add_argument("--lexicon", default=os.path.join(ROOT, "client", "assets", "lexicon.json"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

lexicon = {}
if os.path.exists(args.lexicon):
    lexicon = {e["gloss"]: e["clip"] for e in json.load(open(args.lexicon, encoding="utf-8"))}
used = set(lexicon.values())
n_new = n_skip = 0
with zipfile.ZipFile(args.zip) as z:
    names = sorted(n for n in z.namelist() if n.endswith(".motion.json"))
    for i, name in enumerate(names):
        word = os.path.basename(name)[: -len(".motion.json")]
        word = ALIAS.get(word, word)
        m = json.loads(z.read(name).decode("utf-8"))
        word = m.get("label") or word if word not in MAP else word
        clip = MAP.get(word) or lexicon.get(word)
        if not clip:
            wid = re.search(r"WORD(\d+)", m.get("source", ""))
            clip = f"KSL_W{wid.group(1)}" if wid else f"KSL_G{i:04d}"
            while clip in used and lexicon.get(word) != clip:
                clip += "b"
        if word in lexicon and lexicon[word] == clip and os.path.exists(os.path.join(args.out, f"{clip}.motion.json")):
            n_skip += 1
            continue
        used.add(clip)
        lexicon[word] = clip
        m["clip"] = clip
        m["label"] = word
        with open(os.path.join(args.out, f"{clip}.motion.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, separators=(",", ":"))
        n_new += 1

os.makedirs(os.path.dirname(args.lexicon), exist_ok=True)
with open(args.lexicon, "w", encoding="utf-8") as f:
    json.dump([{"gloss": g, "clip": c} for g, c in sorted(lexicon.items())], f, ensure_ascii=False, indent=0)
print(f"new {n_new}, unchanged {n_skip}, lexicon {len(lexicon)} words -> {args.lexicon}")
