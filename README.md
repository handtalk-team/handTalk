# ✋ handTalk — AI 실시간 수어 튜터 플랫폼

> 스마트 글러브 + 카메라 비전 하이브리드 수어 인식 → LLM 대화 → 실사형 아바타 수어 응답

---

## 프로젝트 개요

handTalk는 **병원 환경에서 청각장애인과 의료진 사이의 소통을 돕는 AI 수어 튜터 플랫폼**입니다.

| 기능 | 설명 |
|------|------|
| 하이브리드 수어 인식 | MediaPipe(카메라) + Flex/IMU 센서(글러브) fusion |
| LLM 수어 튜터 | Claude API 기반 문맥 인식 한국어 대화 |
| 아바타 수어 응답 | MetaHuman / UE5 애니메이션 커맨드 생성 |
| 피드백 엔진 | DTW 기반 동작 비교 + 자동 오답노트 |

---

## 아키텍처

```
[웹 클라이언트]
  ├── 노트북 웹캠  → MediaPipe JS → VisionData (21 landmarks)
  └── 글러브 Mock → GloveMock JS → GloveData (flex 5ch + IMU 9ch)
        │
        │  WebSocket (30 Hz JSON)
        ▼
[FastAPI 서버]
  ├── SensorFusionModule   타임스탬프 정렬 + 77-D feature fusion
  ├── HybridRecognitionEngine  ONNX BiGRU+Attention (fallback: rule-based)
  ├── LLMPipeline          Claude API + 프롬프트 캐싱
  ├── AvatarController     Korean text → MetaHuman 애니메이션 커맨드
  └── FeedbackEngine       DTW + 손가락별 오차 분석
        │
        │  WebSocket (JSON)
        ▼
[웹 클라이언트 UI]
  ├── 수어 인식 결과 표시
  ├── LLM 응답 채팅 UI
  ├── 실시간 피드백
  └── 세션 통계
```

---

## 빠른 시작

### 1. 서버 실행 (로컬)

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env에서 ANTHROPIC_API_KEY 입력

uvicorn app.main:app --reload --port 8000
```

### 2. 웹 클라이언트 실행

```bash
# VS Code Live Server 또는 Python 간이 서버 사용
cd client
python -m http.server 5500
```
브라우저에서 `http://localhost:5500` 접속

### 3. Docker Compose로 전체 스택 실행

```bash
cp server/.env.example server/.env
# .env 파일 편집 후

docker compose up --build
```

---

## 데이터 수집 및 모델 훈련

```bash
# 1. 데이터 수집 (SPACE: 녹화 시작/종료, Q: 종료)
python -m ml.training.collect --label 안녕하세요 --samples 300
python -m ml.training.collect --label 아프다 --samples 300
# ... 10개 수어 모두 수집

# 2. 훈련 (BiGRU + Self-Attention)
python -m ml.training.train --export

# 3. 생성된 모델 확인
ls ml/models/sign_recognizer.onnx
```

---

## 글러브 하드웨어 연결 (ESP32)

1. `firmware/esp32/` 빌드 및 플래시:
   ```bash
   cd firmware/esp32
   idf.py set-target esp32
   idf.py build flash
   ```

2. 서버 `.env` 수정:
   ```
   USE_MOCK_GLOVE=false
   ```

3. 웹 클라이언트 `src/app.js`에서:
   ```js
   // GloveMock → BLEGlove로 교체
   import { BLEGlove } from './glove_mock.js';
   const glove = new BLEGlove();
   ```

---

## 프로젝트 구조

```
handTalk/
├── server/
│   ├── app/
│   │   ├── main.py                    # FastAPI 진입점
│   │   ├── core/                      # config, DB, Redis
│   │   ├── api/
│   │   │   ├── ws/session.py          # WebSocket 세션 핸들러
│   │   │   └── routes/                # REST 엔드포인트
│   │   ├── services/
│   │   │   ├── sync/fusion.py         # 센서 퓨전 + 타임스탬프 동기화
│   │   │   ├── recognition/engine.py  # 하이브리드 인식 엔진
│   │   │   ├── recognition/glove.py   # MockGloveSensor / BLEGloveSensor
│   │   │   ├── recognition/vision.py  # 웹캠 캡처 / 클라이언트 랜드마크
│   │   │   ├── llm/pipeline.py        # Claude API 대화 파이프라인
│   │   │   ├── avatar/controller.py   # 텍스트 → 아바타 커맨드
│   │   │   └── feedback/engine.py     # DTW 피드백 엔진
│   │   └── models/                    # Pydantic 스키마 + SQLAlchemy ORM
│   ├── requirements.txt
│   └── Dockerfile
├── client/
│   ├── index.html                     # 통합 UI
│   └── src/
│       ├── app.js                     # 메인 진입점
│       ├── camera.js                  # 웹캠 + MediaPipe JS
│       ├── glove_mock.js              # GloveMock + BLEGlove 스텁
│       ├── websocket_client.js        # WS 클라이언트
│       └── ui.js                      # DOM 업데이트
├── firmware/esp32/                    # ESP32 BLE 펌웨어
├── ml/
│   └── training/
│       ├── model.py                   # BiGRU + Self-Attention
│       ├── dataset.py                 # 데이터셋 + 증강
│       ├── train.py                   # 훈련 스크립트
│       └── collect.py                 # 데이터 수집 도구
└── docker-compose.yml
```

---

## 성능 목표

| 지표 | 목표 | 현황 |
|------|------|------|
| 수어 인식 정확도 | ≥ 90% (MVP) | 훈련 데이터 수집 필요 |
| 응답 지연 | ≤ 2초 (end-to-end) | 아키텍처 설계 완료 |
| 카메라 처리 | ≥ 30 fps | WebSocket 프레임 처리 |
| Fallback 동작 | 단일 모달 저하 시 자동 전환 | 구현 완료 |

---

## 팀

| 이름 | 역할 |
|------|------|
| 오성현 (팀장) | PM, LLM 프롬프트, 통합 테스트 |
| 김동성 | AI/백엔드, 인식 모델, 서버, 아바타, 피드백 |
| 김경환 | 하드웨어, ESP32 펌웨어, BLE, 센서 캘리브레이션 |
