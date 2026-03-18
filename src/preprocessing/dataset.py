"""
dataset.py
==========
PyTorch Dataset / DataLoader 인터페이스
  - validate_coco     : Letterbox JSON 무결성 검증
  - build_df_from_json: COCO JSON → Pandas DataFrame
  - OralDrugDataset   : Faster R-CNN / RetinaNet용 Dataset 클래스
  - get_loaders       : train/val DataLoader 빌더

🚨 레이블 규칙
  Faster R-CNN / RetinaNet : 1-based  (0 = background 예약)
  YOLO                     : 0-based  (format_converter.py 참고)
"""

import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ImageNet 사전학습 백본 정규화 상수 (추론/시각화 시 활용)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 역정규화 유틸 (추론 / 시각화 시 사용)
# ---------------------------------------------------------------------------
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Normalized 텐서를 시각화 가능한 형태로 복원합니다.
    (주의: DataLoader에서 Normalize를 제거했더라도, 모델 출력물 분석 시 필요할 수 있음)
    """
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


# ---------------------------------------------------------------------------
# 파이프라인 검증
# ---------------------------------------------------------------------------
def validate_coco(json_path, target_size=800):
    """
    Letterbox 처리가 완료된 COCO JSON의 무결성을 검증합니다.
    """
    if not os.path.exists(json_path):
        print(f"🚨 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    wrong_size = [img for img in coco['images']
                  if img['width'] != target_size or img['height'] != target_size]

    issues = []
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        if x < 0 or y < 0:
            issues.append(('negative_xy', ann['id']))
        if w <= 0 or h <= 0:
            issues.append(('non_positive_wh', ann['id']))
        if x + w > target_size or y + h > target_size:
            issues.append(('out_of_bounds', ann['id']))

    print(f"\n[{os.path.basename(json_path)}]")
    print(f"  • 이미지 수        : {len(coco['images'])}장")
    print(f"  • BBox 총 수       : {len(coco['annotations'])}개")
    print(f"  • 규격 이상 이미지  : {len(wrong_size)}장")
    print(f"  • BBox 좌표 이슈   : {len(issues)}개")


# ---------------------------------------------------------------------------
# JSON → DataFrame
# ---------------------------------------------------------------------------
def build_df_from_json(json_path, img_dir):
    """
    COCO JSON을 읽어 annotation 단위의 Pandas DataFrame으로 변환합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_to_fname = {img['id']: img['file_name'] for img in data['images']}
    records = []
    for ann in data['annotations']:
        file_name = id_to_fname.get(ann['image_id'])
        if not file_name:
            continue
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue
        x, y, w, h = ann['bbox']
        records.append({
            'image_path':  img_path,
            'image_id':    os.path.splitext(file_name)[0],
            'category_id': int(ann['category_id']),
            'bbox_x': float(x), 'bbox_y': float(y),
            'bbox_w': float(w), 'bbox_h': float(h),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset 클래스
# ---------------------------------------------------------------------------
class OralDrugDataset(Dataset):
    """
    Faster R-CNN / RetinaNet 학습을 위한 PyTorch Dataset 클래스.
    """

    def __init__(self, df, orig2model, transforms=None):
        self.df         = df.reset_index(drop=True)
        self.orig2model = orig2model
        self.transforms = transforms
        self.image_ids  = self.df['image_id'].unique().tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df_img   = self.df[self.df['image_id'] == image_id]
        image    = Image.open(df_img['image_path'].iloc[0]).convert('RGB')

        boxes, labels = [], []
        for _, row in df_img.iterrows():
            x1, y1 = row['bbox_x'], row['bbox_y']
            x2, y2 = x1 + row['bbox_w'], y1 + row['bbox_h']
            boxes.append([x1, y1, x2, y2])
            labels.append(self.orig2model.get(int(row['category_id']), 1))

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
        }
        if self.transforms:
            image = self.transforms(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# DataLoader 빌더
# ---------------------------------------------------------------------------
def get_loaders(base_dir, batch_size=2, num_workers=2):
    """
    train / val DataLoader를 한 번에 생성하여 반환합니다.
    """
    train_json = os.path.join(base_dir, 'train_letterbox.json')
    val_json   = os.path.join(base_dir, 'val_letterbox.json')
    train_img  = os.path.join(base_dir, 'letterbox_images/train')
    val_img    = os.path.join(base_dir, 'letterbox_images/val')

    df_train = build_df_from_json(train_json, train_img)
    df_val   = build_df_from_json(val_json,   val_img)

    # 레이블 매핑 (0 = background 예약 → 1-based)
    unique_cats = sorted(df_train['category_id'].unique())
    orig2model  = {cid: i + 1 for i, cid in enumerate(unique_cats)}
    num_classes = len(unique_cats) + 1

    print(f"✅ 고유 클래스 수  : {len(unique_cats)}종")
    print(f"✅ num_classes     : {num_classes}  ← 모델 정의 시 사용")
    print(f"✅ Train: {df_train['image_id'].nunique()}장 / {len(df_train)}개")
    print(f"✅ Val  : {df_val['image_id'].nunique()}장 / {len(df_val)}개")

    # 🚨 [수정 완료] T.Normalize 제거 (Faster R-CNN / RetinaNet 내부에서 수행됨)
    train_transforms = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.ToTensor(),
    ])

    train_ds = OralDrugDataset(df_train, orig2model, train_transforms)
    val_ds   = OralDrugDataset(df_val,   orig2model, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, orig2model, num_classes