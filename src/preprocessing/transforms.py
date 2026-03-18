"""
transforms.py
=============
이미지 전처리 핵심 엔진

  - letterbox_with_bbox   : 비율 유지 리사이즈 + BBox 좌표 동기화
  - apply_clahe_to_folder : L-channel CLAHE 대비 강화 (in-place)
  - run_letterbox_pipeline: COCO JSON 전체를 Letterbox 규격으로 일괄 변환 + JSON 갱신

파이프라인 실행 순서
  [Step 1] run_letterbox_pipeline  → letterbox_images/ + *_letterbox.json 생성
  [Step 2] apply_clahe_to_folder   → letterbox_images/ 이미지에 CLAHE in-place 적용
"""

import cv2
import json
import numpy as np
import os
import glob
from collections import defaultdict
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Step 1-A. 핵심 변환 엔진 (단일 이미지 + BBox)
# ---------------------------------------------------------------------------
def letterbox_with_bbox(image, bboxes, target_size=800):
    """
    이미지의 원본 비율(Aspect Ratio)을 유지한 채 target_size로 리사이즈하고,
    남는 공간을 회색(114, 114, 114) 패딩으로 채웁니다. BBox 좌표도 동기화됩니다.

    Args:
        image      : OpenCV BGR 이미지 (np.ndarray)
        bboxes     : COCO 형식 BBox 리스트 [[x, y, w, h], ...]
        target_size: 출력 이미지 해상도 (정사각형, 기본 800)

    Returns:
        padded     : Letterbox 적용된 이미지
        new_bboxes : 변환된 BBox 리스트 (크기 0인 박스는 None으로 반환)
    """
    h, w = image.shape[:2]

    # 긴 쪽 기준 스케일 결정
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 정중앙 배치를 위한 패딩 계산
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    # 114: ImageNet 평균 픽셀값과 유사 → CNN 백본에 최적화
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,  target_size - new_h - pad_top,
        pad_left, target_size - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    new_bboxes = []
    for x, y, bw, bh in bboxes:
        nx  = x  * scale + pad_left
        ny  = y  * scale + pad_top
        nbw = bw * scale
        nbh = bh * scale

        # 경계 클리핑 (부동소수점 오차 방지)
        nx  = max(0.0, min(float(target_size), nx))
        ny  = max(0.0, min(float(target_size), ny))
        nbw = min(target_size - nx, nbw)
        nbh = min(target_size - ny, nbh)

        if nbw > 1.0 and nbh > 1.0:
            new_bboxes.append([round(nx, 2), round(ny, 2), round(nbw, 2), round(nbh, 2)])
        else:
            new_bboxes.append(None)  # 유효하지 않은 박스 → 호출부에서 스킵

    return padded, new_bboxes


# ---------------------------------------------------------------------------
# Step 1-B. 파이프라인 오케스트레이터 (COCO JSON 전체 일괄 처리)
# ---------------------------------------------------------------------------
def run_letterbox_pipeline(json_path, out_json_path, img_out_dir,
                           base_dir=None, target_size=800, desc='Letterbox 변환'):
    """
    COCO JSON을 읽어 전체 이미지를 Letterbox 규격으로 일괄 변환하고
    새로운 JSON(좌표 갱신)과 이미지를 저장합니다.

    NB03의 process_pipeline() 로직과 동일합니다.
    새 데이터셋이 추가되거나 재처리가 필요할 때 노트북 없이 이 함수를 직접 호출하세요.

    Args:
        json_path      : 입력 COCO JSON 경로 (train_augmented_final.json 또는 val.json)
        out_json_path  : 출력 COCO JSON 저장 경로 (*_letterbox.json)
        img_out_dir    : 변환된 이미지를 저장할 폴더 (letterbox_images/train 또는 val)
        base_dir       : 이미지 파일 탐색 루트 (None이면 json_path 상위 폴더 사용)
        target_size    : 출력 해상도 (기본 800, 정사각형)
        desc           : tqdm 진행바 레이블

    Example:
        from src.preprocessing.transforms import run_letterbox_pipeline

        run_letterbox_pipeline(
            json_path     = f'{BASE_DIR}/train_augmented_final.json',
            out_json_path = f'{BASE_DIR}/train_letterbox.json',
            img_out_dir   = f'{BASE_DIR}/letterbox_images/train',
            base_dir      = BASE_DIR,
        )
        run_letterbox_pipeline(
            json_path     = f'{BASE_DIR}/val.json',
            out_json_path = f'{BASE_DIR}/val_letterbox.json',
            img_out_dir   = f'{BASE_DIR}/letterbox_images/val',
            base_dir      = BASE_DIR,
        )
    """
    if not os.path.exists(json_path):
        print(f"🚨 [에러] 파일을 찾을 수 없습니다: {json_path}")
        return

    os.makedirs(img_out_dir, exist_ok=True)

    # 이미지 파일 전체 인덱싱 (한글 경로 / 드라이브 I/O 최적화)
    search_root = base_dir if base_dir else os.path.dirname(json_path)
    all_files = (glob.glob(os.path.join(search_root, '**', '*.png'), recursive=True) +
                 glob.glob(os.path.join(search_root, '**', '*.jpg'), recursive=True) +
                 glob.glob(os.path.join(search_root, '**', '*.JPG'), recursive=True))
    path_map = {os.path.basename(f): f for f in all_files}
    print(f"⚡ 파일 인덱싱 완료: {len(path_map):,}개")

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # O(1) 매칭을 위한 annotation 해시맵
    id_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        id_to_anns[ann['image_id']].append(ann)

    new_images, new_annotations = [], []
    skipped_bbox = 0

    for img_info in tqdm(coco['images'], desc=desc):
        f_name = os.path.basename(img_info['file_name'])
        if f_name not in path_map:
            continue

        img_array = np.fromfile(path_map[f_name], np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue

        anns   = id_to_anns.get(img_info['id'], [])
        bboxes = [ann['bbox'] for ann in anns]

        lb_img, transformed_bboxes = letterbox_with_bbox(img, bboxes, target_size)

        # 이미지 저장
        save_name = f"lb_{img_info['id']:06d}.jpg"
        save_path = os.path.join(img_out_dir, save_name)
        result, enc = cv2.imencode('.jpg', lb_img)
        if result:
            with open(save_path, 'w+b') as f:
                enc.tofile(f)

        # JSON 이미지 메타데이터 갱신
        new_images.append({
            'id':        img_info['id'],
            'file_name': save_name,
            'width':     target_size,
            'height':    target_size,
        })

        # JSON annotation 갱신 (BBox 좌표 + area)
        for ann, final_bbox in zip(anns, transformed_bboxes):
            if final_bbox is None:
                skipped_bbox += 1
                continue
            new_ann = ann.copy()
            new_ann['bbox'] = final_bbox
            new_ann['area'] = round(final_bbox[2] * final_bbox[3], 2)
            new_annotations.append(new_ann)

    # JSON 직렬화 (indent 제거 → 속도 최적화)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'images':      new_images,
            'annotations': new_annotations,
            'categories':  coco['categories'],
        }, f, ensure_ascii=False)

    print(f"✅ [{os.path.basename(out_json_path)}] 저장 완료")
    print(f"   ▶ 이미지 {len(new_images):,}장 / 객체 {len(new_annotations):,}개")
    if skipped_bbox > 0:
        print(f"   ▶ 🗑️  유효하지 않은 BBox {skipped_bbox:,}개 필터링됨")


# ---------------------------------------------------------------------------
# Step 2. L-channel CLAHE 대비 강화 (in-place)
# ---------------------------------------------------------------------------
def apply_clahe_to_folder(folder_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    지정된 폴더의 모든 .jpg 이미지에 L-channel CLAHE를 적용합니다 (in-place 덮어쓰기).

    RGB 왜곡 방지를 위해 BGR → LAB 색공간 변환 후 밝기(L) 채널에만 평활화 적용.
    한글 경로 및 네트워크 드라이브 I/O 에러를 방지하기 위해 np.fromfile 방식 사용.

    ⚠️  run_letterbox_pipeline() 실행 후에 호출해야 합니다.
        letterbox_images/train 및 letterbox_images/val 폴더를 대상으로 실행하세요.

    Args:
        folder_path    : CLAHE를 적용할 이미지 폴더 경로
        clip_limit     : 대비 제한 임계값 (기본 2.0, 높을수록 노이즈 증폭 위험)
        tile_grid_size : 국소 히스토그램 타일 크기 (기본 (8, 8))

    Example:
        from src.preprocessing.transforms import apply_clahe_to_folder

        apply_clahe_to_folder(f'{BASE_DIR}/letterbox_images/train')
        apply_clahe_to_folder(f'{BASE_DIR}/letterbox_images/val')
    """
    if not os.path.exists(folder_path):
        print(f"🚨 [에러] 폴더를 찾을 수 없습니다: {folder_path}")
        return

    img_list = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not img_list:
        print(f"⚠️  [{os.path.basename(folder_path)}] 처리할 이미지가 없습니다.")
        return

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    print(f"🔬 [{os.path.basename(folder_path)}] CLAHE 적용 중 (총 {len(img_list):,}장)")

    for img_path in tqdm(img_list, desc='CLAHE'):
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_merged = cv2.merge((l_clahe, a, b))
        final_img = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

        result, encoded = cv2.imencode('.jpg', final_img)
        if result:
            with open(img_path, mode='w+b') as f:
                encoded.tofile(f)
