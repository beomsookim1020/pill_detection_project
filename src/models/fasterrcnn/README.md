# Faster R-CNN 경구약제 객체 탐지

## 모델 개요
- **모델**: Faster R-CNN + MobileNetV3 Large FPN
- **목적**: 경구약제 이미지에서 약을 바운딩 박스로 검출
- **클래스 수**: 73종 (background 포함 74)

## 데이터셋
| 구분 | 이미지 수 | 바운딩 박스 수 |
|------|---------|-------------|
| Train | 1,800장 | 6,189개 |
| Val | 139장 | 431개 |
| Test | 843장 | - |

## 실험 결과
| 실험 | mAP@50 | 학습 시간 |
|------|--------|---------|
| v1 베이스라인 (ResNet50) | 0.893 | 28분 |
| v2 MobileNetV3 적용 | 0.910 | 7~11분 |
| v2 + Optuna trial 10 | 0.9166 | 7~11분 |
| v2 + Optuna trial 20 | 0.9190 | 7~11분 |
| **v2 Final** | **0.9202** | **7~11분** |

## 주요 개선사항
| 항목 | v1 베이스라인 | v2 Final |
|------|-------------|---------|
| Backbone | ResNet50 FPN | MobileNetV3 Large FPN |
| Scheduler | StepLR | CosineAnnealingLR |
| Early Stopping | ❌ | ✅ mAP@50 기준 (patience=5) |
| Gradient Clipping | ❌ | ✅ max_norm=5.0 |
| Weight Decay | ❌ | ✅ 0.000187 |
| Learning Rate | 1e-4 (고정) | 0.000129 (Optuna) |
| 랜덤 시드 | ❌ | ✅ seed=42 고정 |

## Optuna 하이퍼파라미터 튜닝
- **탐색 파라미터**: lr, weight_decay, eta_min
- **Trial 수**: 20번
- **Pruner**: MedianPruner

| 항목 | trial 10 | trial 20 (최종) |
|------|----------|----------------|
| lr | 0.000248 | 0.000129 |
| weight_decay | 0.000006 | 0.000187 |
| eta_min | 0.0000003 | 0.000058 |
| mAP@50 | 0.9166 | 0.9190 |

## 추론 방식
- **이미지 전처리**: Letterbox 변환 (976×1280 → 800×800)
- **박스 좌표**: 원본 이미지 좌표로 복원
- **score_threshold**: 0.3
- **TOP_K per image**: 4

## 파일 구조
```
fasterrcnn/
├── fasterrcnn_v2.ipynb  # 학습 및 추론 코드
└── README.md
```
