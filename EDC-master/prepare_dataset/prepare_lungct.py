import os
import argparse
import shutil
from collections import Counter

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


def collect_images(root):
    images = []
    for fname in os.listdir(root):
        if is_image(fname):
            images.append(os.path.join(root, fname))
    return images


def safe_copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main(source_dir, target_dir):
    """
    source_dir:
        Lung CT Dataset/Lung CT Dataset/

    target_dir:
        LUNG_CT/
    """

    splits = ["train", "test", "valid"]

    target_train_normal = os.path.join(target_dir, "train", "NORMAL")
    target_test_normal = os.path.join(target_dir, "test", "NORMAL")
    target_test_abnormal = os.path.join(target_dir, "test", "ABNORMAL")

    os.makedirs(target_train_normal, exist_ok=True)
    os.makedirs(target_test_normal, exist_ok=True)
    os.makedirs(target_test_abnormal, exist_ok=True)

    stats = Counter()

    for split in splits:
        split_dir = os.path.join(source_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            images = collect_images(cls_path)

            # -----------------------------
            # NORMAL
            # -----------------------------
            if cls.lower() == "normal":
                if split == "train":
                    for img in images:
                        safe_copy(img, target_train_normal)
                        stats["train_normal"] += 1
                else:
                    for img in images:
                        safe_copy(img, target_test_normal)
                        stats["test_normal"] += 1

            # -----------------------------
            # ABNORMAL (all cancers)
            # -----------------------------
            else:
                for img in images:
                    safe_copy(img, target_test_abnormal)
                    stats["test_abnormal"] += 1

    print("\n===== Lung CT Dataset Preparation Summary =====")
    print(f"Train NORMAL:   {stats['train_normal']}")
    print(f"Test NORMAL:    {stats['test_normal']}")
    print(f"Test ABNORMAL:  {stats['test_abnormal']}")
    print("=============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Lung CT dataset for EDC anomaly detection")
    parser.add_argument(
        "--data-folder",
        default="/content/Lung_CT",
        type=str,
        required=True,
        help="Path to 'Lung CT Dataset/Lung CT Dataset'"
    )
    parser.add_argument(
        "--save-folder",
        default="/content/Lung_CT",
        type=str,
        required=True,
        help="Target folder (e.g., LUNG_CT)"
    )

    args = parser.parse_args()

    main(args.data_folder, args.save_folder)
