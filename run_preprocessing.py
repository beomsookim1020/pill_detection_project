"""
run_preprocessing.py
====================
HealthEat 데이터 전처리 파이프라인 전체 실행 스크립트

파이프라인 순서:
    [Step 1]   Stratified Split       → train_raw.json / val.json
    [Step 1-B] 소수 클래스 스티커 추출 → crops_minority/ + crop_meta.csv
    [Step 2]   Copy-Paste 증강        → train_augmented_final.json
    [Step 3]   Letterbox 800×800 변환 → train_letterbox.json / val_letterbox.json
                                        letterbox_images/train, val/
    [Step 4]   CLAHE 대비 강화        → letterbox_images/ in-place 덮어쓰기
    [Step 5]   YOLO 라벨 변환         → yolo_labels/train, val/ + data.yaml
"""

import os
import sys
import json
import random
from collections import defaultdict

# ============================================================
# [경로 설정] Colab / 로컬 환경 자동 감지
# ============================================================
# ✅ 수정: get_ipython() 방식 → os.path.exists('/content/drive') 방식으로 변경
#
# 기존 방식의 문제:
#   get_ipython()은 실행 클라이언트(VSCode / Jupyter)에 의존하기 때문에
#   VSCode + Colab 커널 연결 환경에서는 Colab 서버에서 실행 중이어도
#   'google.colab' 문자열이 잡히지 않아 로컬로 오감지됩니다.
#
# 수정 후:
#   /content/drive 폴더는 Colab 서버에만 존재하는 경로입니다.
#   실행 클라이언트(VSCode / Jupyter / 터미널)와 무관하게
#   서버 환경만으로 Colab 여부를 정확히 판단합니다.
is_colab = os.path.exists('/content/drive')

if is_colab:
    # ── Colab 환경 ──────────────────────────────────────────
    # VSCode + Colab 커널 환경에서는 드라이브가 이미 마운트되어 있을 수 있으므로
    # try-except로 처리합니다. (중복 마운트 시 그냥 넘어감)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass  # 이미 마운트됐거나 VSCode 커널 환경 → 무시하고 진행

    # 레포가 없으면 자동으로 클론합니다. (최초 1회)
    REPO_DIR = '/content/pill_detection_project'
    if not os.path.exists(REPO_DIR):
        os.system('git clone https://github.com/wina0901/pill_detection_project.git ' + REPO_DIR)

    # 레포 내부에 같은 이름의 중첩 폴더가 있는 경우 대응
    # 예: /content/pill_detection_project/pill_detection_project/src/
    PROJECT_ROOT = REPO_DIR
    nested = os.path.join(REPO_DIR, 'pill_detection_project')
    if os.path.isdir(os.path.join(nested, 'src')):
        PROJECT_ROOT = nested

    # src/preprocessing 등을 import할 수 있도록 실제 루트를 경로에 추가합니다.
    sys.path.insert(0, PROJECT_ROOT)

    # 팀 공통 구글 드라이브 경로 (강사님과 세팅한 경로 그대로 사용)
    BASE_DIR = '/content/drive/MyDrive/data/초급_프로젝트/dataset'

else:
    # ── 로컬 환경 (Mac / Windows) ───────────────────────────
    # __file__ = 현재 이 파일(run_preprocessing.py)의 절대 경로
    # os.path.dirname(__file__) = 이 파일이 있는 폴더 (= 프로젝트 루트)
    #
    # 예시:
    #   이 파일 위치: ~/Desktop/pill_detection_project/run_preprocessing.py
    #   PROJECT_ROOT: ~/Desktop/pill_detection_project
    #   BASE_DIR    : ~/Desktop/pill_detection_project/data
    #
    # ✅ 폴더명이 pill_detection_project_team이어도 괜찮아요!
    #    run_preprocessing.py 파일이 있는 폴더를 기준으로 자동으로 잡아줍니다.
    #    절대 직접 경로를 수정해서 git push 하지 마세요!
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, PROJECT_ROOT)
    BASE_DIR = os.path.join(PROJECT_ROOT, 'data')

print(f"✅ 환경: {'Colab' if is_colab else '로컬'}")
print(f"✅ BASE_DIR: {BASE_DIR}")

# 경로 설정이 끝난 후 모듈을 import합니다.
# (sys.path에 경로가 추가된 다음에 import해야 정상 동작)
from src.preprocessing.augmentation     import extract_minority_crops, run_copy_paste
from src.preprocessing.transforms       import run_letterbox_pipeline, apply_clahe_to_folder
from src.preprocessing.format_converter import run_yolo_conversion


# ============================================================
# Step 1. Stratified Split (9:1)
# ============================================================
def run_stratified_split(base_dir, val_ratio=0.1, random_seed=42):
    """
    원본 COCO JSON을 클래스 비율을 보존하며 Train / Val 로 분할합니다.

    Args:
        base_dir    : merged_annotations_train_final.json 이 있는 데이터 루트
        val_ratio   : Validation 비율 (기본 0.1 = 10%)
        random_seed : 재현성 시드 (기본 42)

    출력:
        base_dir/train_raw.json
        base_dir/val.json
    """
    original_json = os.path.join(base_dir, 'merged_annotations_train_final.json')
    train_out     = os.path.join(base_dir, 'train_raw.json')
    val_out       = os.path.join(base_dir, 'val.json')

    if not os.path.exists(original_json):
        raise FileNotFoundError(f"🚨 원본 JSON 없음: {original_json}")

    print(f"\n{'='*60}")
    print(f"[Step 1] Stratified Split ({int((1-val_ratio)*100)}:{int(val_ratio*100)})")
    print(f"{'='*60}")

    random.seed(random_seed)  # 재현성 보장: 항상 동일한 순서로 섞임

    with open(original_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images, annotations, categories = coco['images'], coco['annotations'], coco['categories']

    # 이미지별 대표 클래스(최빈 클래스) 추출
    # 한 이미지에 여러 알약이 있을 경우, 가장 많이 등장한 클래스를 대표로 선정
    img_to_cats = defaultdict(list)
    for ann in annotations:
        img_to_cats[ann['image_id']].append(ann['category_id'])

    img_dominant = {
        img_id: max(set(cats), key=cats.count)
        for img_id, cats in img_to_cats.items()
    }

    # 클래스별 이미지 목록 구성
    class_to_imgs = defaultdict(list)
    for img_id, label in img_dominant.items():
        class_to_imgs[label].append(img_id)

    train_ids, val_ids = set(), set()

    for label, img_list in class_to_imgs.items():
        random.shuffle(img_list)  # 시드 고정으로 항상 동일하게 섞임

        if len(img_list) == 1:
            # 데이터가 1장뿐인 클래스 → Train에만 넣음 (Val에 넣으면 학습 불가)
            train_ids.update(img_list)
        elif len(img_list) < 5:
            # 소수 클래스 → Val에 최소 1장 보장, 나머지는 Train
            val_ids.add(img_list[0])
            train_ids.update(img_list[1:])
        else:
            # 일반 클래스 → 정확히 9:1 비율로 분할
            split_idx = max(1, int(len(img_list) * val_ratio))
            val_ids.update(img_list[:split_idx])
            train_ids.update(img_list[split_idx:])

    # 분할 결과 검증
    val_classes = set(img_dominant[i] for i in val_ids if i in img_dominant)
    missing     = set(img_dominant.values()) - val_classes
    print(f"📊 총 {len(images):,}장 → Train: {len(train_ids):,}장 / Val: {len(val_ids):,}장")
    if missing:
        # 데이터가 1장뿐인 클래스는 Train에만 들어가므로 Val 누락은 정상
        print(f"⚠️  Val 누락 클래스 {len(missing)}개 (데이터 1장뿐인 클래스 — 정상)")
    else:
        print(f"✅ 전체 {len(categories)}개 클래스 Val 포함 확인")

    # JSON 저장
    train_anns = [a for a in annotations if a['image_id'] in train_ids]
    val_anns   = [a for a in annotations if a['image_id'] in val_ids]

    for path, imgs, anns in [
        (train_out, [i for i in images if i['id'] in train_ids], train_anns),
        (val_out,   [i for i in images if i['id'] in val_ids],   val_anns),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'images': imgs, 'annotations': anns, 'categories': categories},
                      f, ensure_ascii=False)

    print(f"✅ 저장 완료 → {os.path.basename(train_out)} ({len(train_anns):,}개) / "
          f"{os.path.basename(val_out)} ({len(val_anns):,}개)")


# ============================================================
# 메인 파이프라인
# ============================================================
def main():
    print(f"\n{'#'*60}")
    print(f"  HealthEat 데이터 전처리 파이프라인 시작")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"{'#'*60}")

    # ── Step 1. Train/Val 분리 (9:1 Stratified Split)
    # merged_annotations_train_final.json → train_raw.json / val.json
    run_stratified_split(base_dir=BASE_DIR)

    # ── Step 1-B. 소수 클래스 스티커 추출
    # train_raw.json에서 50개 미만 클래스 객체를 잘라서 저장
    # → crops_minority/ 폴더 + crop_meta.csv 생성
    # ⚠️ 이 단계가 없으면 Step 2(Copy-Paste)에서 오류 발생
    extract_minority_crops(base_dir=BASE_DIR, threshold=50)

    # ── Step 2. Copy-Paste 증강
    # 소수 클래스 스티커를 다른 이미지 빈 공간에 합성
    # → 4,095개 → 6,199개로 증가
    print(f"\n{'='*60}")
    print(f"[Step 2] Copy-Paste 증강")
    print(f"{'='*60}")
    run_copy_paste(base_dir=BASE_DIR, aug_count=500, random_seed=42)

    # ── Step 3. Letterbox 800×800 규격화
    # 모든 이미지를 비율 유지하며 800×800으로 통일 (여백은 회색으로 채움)
    # BBox 좌표도 함께 변환 및 클리핑
    print(f"\n{'='*60}")
    print(f"[Step 3] Letterbox 규격화 (800×800)")
    print(f"{'='*60}")
    run_letterbox_pipeline(
        json_path     = os.path.join(BASE_DIR, 'train_augmented_final.json'),
        out_json_path = os.path.join(BASE_DIR, 'train_letterbox.json'),
        img_out_dir   = os.path.join(BASE_DIR, 'letterbox_images/train'),
        base_dir      = BASE_DIR,
        desc          = 'Train Letterbox 변환',
    )
    run_letterbox_pipeline(
        json_path     = os.path.join(BASE_DIR, 'val.json'),
        out_json_path = os.path.join(BASE_DIR, 'val_letterbox.json'),
        img_out_dir   = os.path.join(BASE_DIR, 'letterbox_images/val'),
        base_dir      = BASE_DIR,
        desc          = 'Val Letterbox 변환',
    )

    # ── Step 4. CLAHE 대비 강화 (in-place 덮어쓰기)
    # 알약 표면 글자(각인)가 잘 보이도록 L-channel 대비 강화
    # letterbox_images/ 폴더의 이미지를 직접 덮어씁니다
    print(f"\n{'='*60}")
    print(f"[Step 4] L-channel CLAHE 대비 강화")
    print(f"{'='*60}")
    apply_clahe_to_folder(os.path.join(BASE_DIR, 'letterbox_images/train'))
    apply_clahe_to_folder(os.path.join(BASE_DIR, 'letterbox_images/val'))

    # ── Step 5. YOLO 라벨 변환 + data.yaml 생성
    # train_letterbox.json / val_letterbox.json → YOLO .txt 포맷으로 변환
    # yolo_labels/train, val/ 폴더 생성 및 data.yaml 자동 생성
    print(f"\n{'='*60}")
    print(f"[Step 5] YOLO 라벨 변환 및 data.yaml 생성")
    print(f"{'='*60}")
    run_yolo_conversion(base_dir=BASE_DIR)

    print(f"\n{'#'*60}")
    print(f"  ✅ 전처리 파이프라인 완료!")
    print(f"  학습 준비 완료 경로: {BASE_DIR}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
