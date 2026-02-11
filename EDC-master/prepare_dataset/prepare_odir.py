import os
import random
import argparse
import cv2
import numpy as np
import pandas as pd

from skimage.measure import regionprops, label
from operator import attrgetter


# --------------------------------------------------
# SAME IMAGE PROCESSING FUNCTIONS (UNCHANGED)
# --------------------------------------------------

def fill_crop(img, min_idx, max_idx):
    crop = np.zeros(np.array(max_idx, dtype='int16') - np.array(min_idx, dtype='int16'), dtype=img.dtype)
    img_shape, start, crop_shape = np.array(img.shape), np.array(min_idx, dtype='int16'), np.array(crop.shape)
    end = start + crop_shape

    crop_low = np.clip(0 - start, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))

    pos = np.clip(start, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))

    crop[tuple(crop_slices)] = img[tuple(img_slices)]
    return crop


def fundus_crop(image, shape=[512, 512], margin=5):
    mask = (image.sum(axis=-1) > 30)
    mask = label(mask)
    regions = regionprops(mask)
    region = max(regions, key=attrgetter('area'))

    length = (np.array(region.bbox[2:4]) - np.array(region.bbox[0:2])).max()
    bbox = np.concatenate([
        np.array(region.centroid) - length / 2,
        np.array(region.centroid) + length / 2
    ]).astype('int16')

    image_b = fill_crop(
        image,
        [bbox[0] - margin, bbox[1] - margin, 0],
        [bbox[2] + margin, bbox[3] + margin, 3]
    )
    image_b = cv2.resize(image_b, shape, interpolation=cv2.INTER_LINEAR)
    return image_b


# --------------------------------------------------
# ARGUMENT PARSER (MATCHING APTOS STYLE)
# --------------------------------------------------

parser = argparse.ArgumentParser(description='ODIR Data Preparation for EDC')
parser.add_argument('--data-folder', type=str, required=True,
                    help='Path to ODIR-5K/ODIR-5K')
parser.add_argument('--save-folder', type=str, required=True,
                    help='Target folder for EDC-ready ODIR')
args = parser.parse_args()

random.seed(1)

SOURCE_DIR = args.data_folder
TARGET_DIR = args.save_folder

TRAIN_IMG_DIR = os.path.join(SOURCE_DIR, 'Training Images')
XLSX_PATH = os.path.join(SOURCE_DIR, 'data.xlsx')


# --------------------------------------------------
# OUTPUT DIRECTORIES (IDENTICAL CONTRACT)
# --------------------------------------------------

os.makedirs(os.path.join(TARGET_DIR, 'train', 'NORMAL'), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, 'test', 'NORMAL'), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, 'test', 'ABNORMAL'), exist_ok=True)


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

df = pd.read_excel(XLSX_PATH)

train_normal = []
test_normal = []
abnormal = []


# --------------------------------------------------
# APPLY YOUR EXACT 4-CASE LOGIC
# (only change: safer keyword check)
# --------------------------------------------------

for _, r in df.iterrows():

    patient_N = int(r['N'])
    disease_flags = {d: int(r[d]) for d in ['D','G','C','A','H','M','O']}

    for img_col, kw_col in [
        ('Left-Fundus', 'Left-Diagnostic Keywords'),
        ('Right-Fundus', 'Right-Diagnostic Keywords')
    ]:
        fname = r[img_col]
        kw = str(r[kw_col]).lower()

        img_path = os.path.join(TRAIN_IMG_DIR, fname)

        if not os.path.exists(img_path):
            continue

        is_kw_normal = ('normal fundus' in kw)

        # ---------- CASE 1 ----------
        if is_kw_normal and patient_N == 1:
            train_normal.append(img_path)

        # ---------- CASE 4 ----------
        elif (not is_kw_normal) and patient_N == 0:
            abnormal.append(img_path)

        # CASE 2 & 3 → intentionally ignored (excluded)


# --------------------------------------------------
# TRAIN / TEST SPLIT (MATCHING APTOS BEHAVIOR)
# --------------------------------------------------

random.shuffle(train_normal)

TRAIN_NORMAL = train_normal[:1000]
TEST_NORMAL = train_normal[1000:]


# --------------------------------------------------
# SAVE IMAGES (IDENTICAL FLOW)
# --------------------------------------------------

def process_and_save(paths, out_dir):
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = fundus_crop(img, shape=[512, 512], margin=5)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), img)


process_and_save(TRAIN_NORMAL, os.path.join(TARGET_DIR, 'train', 'NORMAL'))
process_and_save(TEST_NORMAL,  os.path.join(TARGET_DIR, 'test', 'NORMAL'))
process_and_save(abnormal,     os.path.join(TARGET_DIR, 'test', 'ABNORMAL'))


print("===== ODIR PREPARATION COMPLETE =====")
print(f"Train NORMAL: {len(TRAIN_NORMAL)}")
print(f"Test NORMAL:  {len(TEST_NORMAL)}")
print(f"Test ABNORMAL:  {len(abnormal)}")
print("====================================")
