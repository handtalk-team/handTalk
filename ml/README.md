# ML — 수어 인식 모델

## 모델: BiGRU + Self-Attention

```
Input (B, T=60, D=77)
    ↓ Linear projection
(B, T, 128)
    ↓ Bidirectional GRU × 2 layers
(B, T, 256)
    ↓ Self-Attention + Residual
(B, T, 256)
    ↓ Global Average Pooling
(B, 256)
    ↓ MLP Classifier
(B, 10)  ← 10개 수어 클래스
```

파라미터 수: ~120K (CPU에서 추론 < 10ms)

## 왜 CNN+LSTM이 아닌가?

- MediaPipe가 이미 공간 특징(랜드마크)을 추출 → CNN 불필요
- GRU: LSTM 대비 파라미터 33% 적어 소규모 데이터(3,000샘플)에 유리
- Self-Attention: 핵심 프레임(제스처 피크)에 자동 집중
- 3,000샘플에서 val accuracy 90%+ 달성 가능 (데이터 증강 적용 시)

## 데이터 수집 가이드

```bash
# 10개 수어 각 300회 녹화
python -m ml.training.collect --label 안녕하세요 --samples 300
python -m ml.training.collect --label 감사합니다 --samples 300
python -m ml.training.collect --label 아프다      --samples 300
python -m ml.training.collect --label 약          --samples 300
python -m ml.training.collect --label 의사        --samples 300
python -m ml.training.collect --label 간호사      --samples 300
python -m ml.training.collect --label 도와주세요  --samples 300
python -m ml.training.collect --label 화장실      --samples 300
python -m ml.training.collect --label 물          --samples 300
python -m ml.training.collect --label 괜찮아요    --samples 300
```

## 훈련

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ml.training.train --export
```

## ONNX 배포

훈련 완료 후 `ml/models/sign_recognizer.onnx`가 생성됩니다.
서버 재시작 시 자동으로 로드됩니다.
