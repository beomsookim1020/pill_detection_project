"""
src/preprocessing/__init__.py
==============================
HealthEat 데이터 전처리 패키지

파이프라인 실행 순서
  [Step 1] transforms.run_letterbox_pipeline  → Letterbox 800×800 변환 + JSON 갱신
  [Step 2] transforms.apply_clahe_to_folder   → L-channel CLAHE 대비 강화 (in-place)
  [Step 3] augmentation.run_copy_paste        → Copy-Paste 소수 클래스 증강
  [Step 4] dataset.get_loaders                → Faster R-CNN / RetinaNet DataLoader
  [Step 4] format_converter.run_yolo_conversion → YOLO 라벨 변환 + data.yaml 생성

모듈 구성
  transforms.py      : letterbox_with_bbox, run_letterbox_pipeline, apply_clahe_to_folder
  augmentation.py    : check_overlap, blend_image, run_copy_paste
  dataset.py         : OralDrugDataset, build_df_from_json, get_loaders, validate_coco, denormalize
  format_converter.py: convert_coco_to_yolo, generate_data_yaml, run_yolo_conversion
"""

from .transforms       import letterbox_with_bbox, run_letterbox_pipeline, apply_clahe_to_folder
from .augmentation     import check_overlap, blend_image, run_copy_paste
from .dataset          import (OralDrugDataset, build_df_from_json,
                                get_loaders, validate_coco, denormalize,
                                collate_fn, IMAGENET_MEAN, IMAGENET_STD)
from .format_converter import (convert_coco_to_yolo, generate_data_yaml,
                                run_yolo_conversion)

__all__ = [
    # transforms
    'letterbox_with_bbox', 'run_letterbox_pipeline', 'apply_clahe_to_folder',
    # augmentation
    'check_overlap', 'blend_image', 'run_copy_paste',
    # dataset
    'OralDrugDataset', 'build_df_from_json', 'get_loaders',
    'validate_coco', 'denormalize', 'collate_fn',
    'IMAGENET_MEAN', 'IMAGENET_STD',
    # format_converter
    'convert_coco_to_yolo', 'generate_data_yaml', 'run_yolo_conversion',
]
