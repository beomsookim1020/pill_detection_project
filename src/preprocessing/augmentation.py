"""
augmentation.py
===============
Copy-Paste 증강 엔진
  - check_overlap  : AABB 충돌 감지 (새 박스가 기존 박스와 겹치는지 검사)
  - blend_image    : 스티커-배경 경계선 자연스럽게 블렌딩
  - run_copy_paste : 전체 증강 파이프라인 실행 (JSON + 이미지 동시 생성)
"""

import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import random
import copy
from tqdm.auto import tqdm
from collections import defaultdict


def check_overlap(new_box, existing_boxes, min_dist=15):
    """
    새로 붙일 알약이 기존 박스들과 겹치는지 AABB 방식으로 검사합니다.

    Args:
        new_box       : (x, y, w, h) 형태의 새 박스
        existing_boxes: [(x, y, w, h), ...] 형태의 기존 박스 목록
        min_dist      : 박스 간 최소 안전 거리 (픽셀, 기본 15)

    Returns:
        True  : 겹침 발생 → 해당 위치 사용 불가
        False : 안전 → 해당 위치 사용 가능
    """
    nx, ny, nw, nh = new_box
    for ex, ey, ew, eh in existing_boxes:
        if not (nx + nw + min_dist < ex or
                nx > ex + ew + min_dist or
                ny + nh + min_dist < ey or
                ny > ey + eh + min_dist):
            return True
    return False


def blend_image(bg_crop, sticker):
    """
    알약 스티커를 배경에 자연스럽게 합성합니다 (Feathering 효과).
    Hard Edge를 모델이 합성 힌트로 학습하지 못하도록 경계선을 부드럽게 처리합니다.

    Args:
        bg_crop : 붙일 위치의 배경 패치 (np.ndarray)
        sticker : 알약 스티커 이미지 (np.ndarray)

    Returns:
        blended : 배경 20% + 스티커 80% 가중 합성 이미지
    """
    if bg_crop.shape != sticker.shape:
        return sticker
    return cv2.addWeighted(bg_crop, 0.2, sticker, 0.8, 0)


def run_copy_paste(base_dir, aug_count=500, random_seed=42):
    """
    Copy-Paste 증강을 실행하여 새로운 합성 이미지와 COCO JSON을 생성합니다.

    사전 조건:
      - base_dir/train_raw.json        : NB02에서 분할된 Train 데이터
      - base_dir/crops_minority/       : NB02에서 추출된 소수 클래스 스티커
      - base_dir/crops_minority/crop_meta.csv : 스티커 메타데이터

    출력:
      - base_dir/train_augmented_images/ : 합성 이미지
      - base_dir/train_augmented_final.json : 증강된 COCO JSON

    Args:
        base_dir    : 데이터셋 루트 경로
        aug_count   : 생성할 합성 이미지 수 (기본 500)
        random_seed : 재현성을 위한 시드 (기본 42)
    """
    random.seed(random_seed)

    train_json_path = os.path.join(base_dir, 'train_raw.json')
    crop_dir        = os.path.join(base_dir, 'crops_minority')
    meta_path       = os.path.join(crop_dir, 'crop_meta.csv')
    aug_img_dir     = os.path.join(base_dir, 'train_augmented_images')
    aug_json_path   = os.path.join(base_dir, 'train_augmented_final.json')
    os.makedirs(aug_img_dir, exist_ok=True)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"스티커 메타데이터를 찾을 수 없습니다: {meta_path}")

    with open(train_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    aug_coco   = copy.deepcopy(coco_data)
    crop_meta  = pd.read_csv(meta_path)

    # 파일 경로 해시맵 (O(1) 탐색)
    all_files = (glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True) +
                 glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True) +
                 glob.glob(os.path.join(base_dir, "**", "*.JPG"), recursive=True))
    path_map = {os.path.basename(f): f for f in all_files}

    max_img_id  = max((img['id']  for img in aug_coco['images']),      default=0)
    max_anno_id = max((ann['id']  for ann in aug_coco['annotations']), default=0)

    bg_candidates = list(coco_data['images'])

    for _ in tqdm(range(aug_count), desc="Copy-Paste 증강"):
        bg_info = random.choice(bg_candidates)
        f_name  = os.path.basename(bg_info['file_name'])
        if f_name not in path_map:
            continue

        bg_array = np.fromfile(path_map[f_name], np.uint8)
        bg_img   = cv2.imdecode(bg_array, cv2.IMREAD_COLOR)
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]
        existing_boxes = [ann['bbox'] for ann in aug_coco['annotations']
                          if ann['image_id'] == bg_info['id']]

        num_pastes   = random.randint(1, 4)
        pastes       = crop_meta.sample(num_pastes, replace=True)
        new_anns     = []
        success      = False

        for _, row in pastes.iterrows():
            cw, ch = int(row['width']), int(row['height'])
            crop_arr = np.fromfile(row['crop_path'], np.uint8)
            crop_img = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
            if crop_img is None:
                continue

            # 빈 자리 탐색 (최대 50회 시도)
            safe_x = safe_y = -1
            for _ in range(50):
                max_x = max(11, bg_w - cw - 10)
                max_y = max(11, bg_h - ch - 10)
                if max_x <= 10 or max_y <= 10:
                    break
                rx, ry = random.randint(10, max_x), random.randint(10, max_y)
                if not check_overlap((rx, ry, cw, ch), existing_boxes):
                    safe_x, safe_y = rx, ry
                    break

            if safe_x != -1 and safe_x + cw <= bg_w and safe_y + ch <= bg_h:
                bg_patch = bg_img[safe_y:safe_y+ch, safe_x:safe_x+cw]
                bg_img[safe_y:safe_y+ch, safe_x:safe_x+cw] = blend_image(bg_patch, crop_img)
                existing_boxes.append([safe_x, safe_y, cw, ch])

                max_anno_id += 1
                new_anns.append({
                    "id":          max_anno_id,
                    "image_id":    max_img_id + 1,
                    "category_id": int(row['category_id']),
                    "bbox":        [safe_x, safe_y, cw, ch],
                    "area":        float(cw * ch),
                    "iscrowd":     0,
                    "segmentation": []
                })
                success = True

        if success:
            max_img_id  += 1
            new_filename = f"aug_cp_{max_img_id:06d}.jpg"
            save_path    = os.path.join(aug_img_dir, new_filename)

            result, encoded = cv2.imencode('.jpg', bg_img)
            if result:
                with open(save_path, mode='w+b') as f:
                    encoded.tofile(f)

            aug_coco['images'].append({
                "id": max_img_id, "file_name": new_filename,
                "width": bg_w,   "height": bg_h
            })

            # 원본 이미지의 기존 annotation 복사
            for ann in aug_coco['annotations']:
                if ann['image_id'] == bg_info['id']:
                    max_anno_id += 1
                    cloned = copy.deepcopy(ann)
                    cloned['id']       = max_anno_id
                    cloned['image_id'] = max_img_id
                    aug_coco['annotations'].append(cloned)

            # 새로 붙인 스티커 annotation 추가
            aug_coco['annotations'].extend(new_anns)

    with open(aug_json_path, 'w', encoding='utf-8') as f:
        json.dump(aug_coco, f, ensure_ascii=False)

    print(f"\n✅ Copy-Paste 증강 완료")
    print(f"   원본 객체: {len(coco_data['annotations']):,}개")
    print(f"   증강 후 : {len(aug_coco['annotations']):,}개")
    print(f"   저장 경로: {aug_json_path}")
