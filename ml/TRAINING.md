# handTalk 학습 가이드

## 환경 설정 (Windows / Linux GPU 머신)

```bash
# 1. 레포 클론
git clone https://github.com/handtalk-team/handTalk.git
cd handTalk

# 2. 가상환경 & 패키지
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install -r server/requirements.txt
```

---

## 데이터 구조 (구글 드라이브에서 받은 것)

```
handTalk_data/
├── 수어 영상/
│   └── 1.Training/
│       ├── 01/          ← keypoint batch 01 (REAL01)
│       ├── 02/          ← keypoint batch 02 (REAL02)
│       │   ...
│       ├── 16/          ← keypoint batch 16
│       └── morpheme/    ← 라벨 JSON
│           ├── 01/
│           │   ...
│           └── 16/
```

---

## Step 1: 키포인트 → npy 추출

```bash
python -m ml.preprocessing.extract_aihub \
  --keypoint "/path/to/수어 영상/1.Training" \
  --morpheme "/path/to/수어 영상/1.Training/morpheme" \
  --output   ml/data \
  --words 병원 아프다 괜찮다 치료 건강 감기 의사 간호사 검사 입원
```

- 배치 01~16 자동 감지해서 순서대로 처리
- 이미 추출된 것은 스킵 (재실행해도 중복 없음)
- 약 800개 클립 → 소요 시간 약 10초

---

## Step 2: 학습

```bash
python -m ml.training.train \
  --data_dir  ml/data \
  --model_dir ml/models \
  --export
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `ml/data` | 추출된 npy 경로 |
| `--model_dir` | `ml/models` | 모델 저장 경로 |
| `--resume` | — | 중단된 학습 이어하기 (`checkpoints/epoch_050.pt`) |
| `--export` | — | 학습 후 ONNX 자동 생성 |
| `--num_workers` | 4 | DataLoader 병렬 로딩 수 |

### GPU별 예상 시간 (800 샘플 기준)

| GPU | 학습 (150에폭) | 비고 |
|-----|--------------|------|
| RTX 3070 | ~5분 | CUDA 12.1 |
| RTX 5090 | ~1~2분 | CUDA 12.4+ |
| CPU only | ~45분 | 권장 안함 |

---

## Step 3: 서버에 모델 반영

```bash
# 학습된 ONNX를 서버 모델 경로로 복사
cp ml/models/sign_recognizer.onnx server/ml/models/sign_recognizer.onnx
```

---

## 전체 한 번에 (Windows)

```batch
set DATA=D:\handTalk_data
set MODEL=D:\handTalk_models

python -m ml.preprocessing.extract_aihub ^
  --keypoint "%DATA%\수어 영상\1.Training" ^
  --morpheme "%DATA%\수어 영상\1.Training\morpheme" ^
  --output   ml\data

python -m ml.training.train ^
  --data_dir ml\data ^
  --model_dir "%MODEL%" ^
  --export
```

## 전체 한 번에 (Linux)

```bash
DATA="/mnt/d/handTalk_data"
python -m ml.preprocessing.extract_aihub \
  --keypoint "$DATA/수어 영상/1.Training" \
  --morpheme "$DATA/수어 영상/1.Training/morpheme" \
  --output   ml/data && \
python -m ml.training.train \
  --data_dir ml/data --model_dir ml/models --export
```
