#!/bin/bash
# ============================================================
# handTalk — 로컬 GPU 학습 스크립트
# 사용법:
#   bash ml/train_gpu.sh [DATA_DIR] [MODEL_DIR]
#
# 예시:
#   bash ml/train_gpu.sh ml/data ml/models
#   bash ml/train_gpu.sh D:/handTalk_data/npy D:/handTalk_models
# ============================================================

DATA_DIR=${1:-"ml/data"}
MODEL_DIR=${2:-"ml/models"}

echo "=================================================="
echo " handTalk GPU 학습"
echo " DATA_DIR  : $DATA_DIR"
echo " MODEL_DIR : $MODEL_DIR"
echo "=================================================="

# 1단계: AI Hub 키포인트 → npy 추출
# (이미 추출했으면 --overwrite 없이 실행하면 스킵됨)
echo ""
echo "[1/2] 데이터 추출..."
python -m ml.preprocessing.extract_aihub \
  --keypoint "${DATA_DIR}/원천/수어 영상/1.Training" \
  --morpheme "${DATA_DIR}/라벨/수어 영상/1.Training/morpheme" \
  --output   "${DATA_DIR}/npy" \
  --words 병원 아프다 괜찮다 치료 건강 감기 의사 간호사 검사 입원

# 2단계: 학습 + ONNX 내보내기
echo ""
echo "[2/2] 학습 시작..."
python -m ml.training.train \
  --data_dir    "${DATA_DIR}/npy" \
  --model_dir   "$MODEL_DIR" \
  --num_workers 4 \
  --export

echo ""
echo "완료! ONNX 모델: ${MODEL_DIR}/sign_recognizer.onnx"
