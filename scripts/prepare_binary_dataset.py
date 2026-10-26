"""
scripts/prepare_binary_dataset.py
Extracts the 323 original images from Mendeley Data archive mhjyrn35p4-2.zip
and creates a reproducible, stratified train/validation/test split for the
two-class (Benign vs Malignant) experimental research mode.

Pre-augmented images are strictly excluded.
"""

import hashlib
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from PIL import Image

ZIP_PATH = Path(r"C:\Users\DELL\Downloads\mhjyrn35p4-2.zip")
OUTPUT_DIR = Path("data_binary")
RANDOM_SEED = 42

def prepare_binary_dataset():
    if not ZIP_PATH.exists():
        print(f"Error: Archive not found at {ZIP_PATH}")
        sys.exit(1)

    print(f"Opening archive: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()

        benign_files = sorted([
            n for n in all_names
            if n.startswith("Oral Images Dataset/original_data/benign_lesions/")
            and not n.endswith("/")
        ])
        malignant_files = sorted([
            n for n in all_names
            if n.startswith("Oral Images Dataset/original_data/malignant_lesions/")
            and not n.endswith("/")
        ])

        print(f"Found {len(benign_files)} original Benign images.")
        print(f"Found {len(malignant_files)} original Malignant images.")

        assert len(benign_files) == 165, f"Expected 165 benign, got {len(benign_files)}"
        assert len(malignant_files) == 158, f"Expected 158 malignant, got {len(malignant_files)}"

        # Verify integrity and compute hashes
        def verify_and_read(file_list, label_name):
            verified = []
            for item in file_list:
                raw = z.read(item)
                sha = hashlib.sha256(raw).hexdigest()
                # Verify valid image
                try:
                    import io
                    with Image.open(io.BytesIO(raw)) as img:
                        img.verify()
                except Exception as e:
                    print(f"Corrupt image encountered: {item} ({e})")
                    continue
                filename = Path(item).name
                verified.append({
                    "archive_path": item,
                    "filename": filename,
                    "label": label_name,
                    "sha256": sha,
                    "raw_bytes": raw
                })
            return verified

        benign_items = verify_and_read(benign_files, "Benign")
        malignant_items = verify_and_read(malignant_files, "Malignant")

        print(f"Verified {len(benign_items)} Benign and {len(malignant_items)} Malignant images.")

        # Reproducible stratified split (70% train, 15% val, 15% test)
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(benign_items)
        rng.shuffle(malignant_items)

        # 165: 115 train, 25 val, 25 test
        b_train = benign_items[:115]
        b_val = benign_items[115:140]
        b_test = benign_items[140:]

        # 158: 110 train, 24 val, 24 test
        m_train = malignant_items[:110]
        m_val = malignant_items[110:134]
        m_test = malignant_items[134:]

        splits = {
            "train": {"Benign": b_train, "Malignant": m_train},
            "val": {"Benign": b_val, "Malignant": m_val},
            "test": {"Benign": b_test, "Malignant": m_test},
        }

        # Create output directories and write files
        manifest = {
            "source_archive": str(ZIP_PATH),
            "random_seed": RANDOM_SEED,
            "classes": ["Benign", "Malignant"],
            "disclaimer": (
                "Patient identifiers were not provided in the source dataset. "
                "Patient-level leakage cannot be ruled out. This benchmark is an "
                "image-level experimental evaluation and not a patient-independent clinical study."
            ),
            "counts": {},
            "files": {}
        }

        for split_name, class_dict in splits.items():
            manifest["counts"][split_name] = {}
            for label, items in class_dict.items():
                split_dir = OUTPUT_DIR / split_name / label
                split_dir.mkdir(parents=True, exist_ok=True)
                manifest["counts"][split_name][label] = len(items)

                for item in items:
                    out_path = split_dir / item["filename"]
                    with open(out_path, "wb") as f:
                        f.write(item["raw_bytes"])

                    manifest["files"][str(out_path.as_posix())] = {
                        "split": split_name,
                        "label": label,
                        "sha256": item["sha256"],
                        "original_archive_path": item["archive_path"]
                    }

        # Save manifest
        manifest_path = OUTPUT_DIR / "split_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save README
        readme_path = OUTPUT_DIR / "README.md"
        readme_content = f"""# ORALNOVA-AI Two-Class Experimental Dataset (Benign vs Malignant)

This dataset was extracted exclusively from the 323 original images in:
`Mendeley Data DOI: 10.17632/mhjyrn35p4.2` (Chandrashekar et al., 2021)

## Class Breakdown
- **Benign**: 165 original images
- **Malignant**: 158 original images
- **Total**: 323 clinical photographs

## Stratified Split (Seed: {RANDOM_SEED})
| Split | Benign | Malignant | Total |
|---|---|---|---|
| Train | {len(b_train)} | {len(m_train)} | {len(b_train) + len(m_train)} |
| Val | {len(b_val)} | {len(m_val)} | {len(b_val) + len(m_val)} |
| Test | {len(b_test)} | {len(m_test)} | {len(b_test) + len(m_test)} |
| **Total** | **165** | **158** | **323** |

## Critical Clinical Research Notice
Patient identifiers were not supplied in the Mendeley Data archive. Therefore, **patient-level leakage cannot be ruled out**. This is an image-level benchmark for algorithm comparison and college demonstration, NOT a patient-independent clinical diagnostic evaluation.

All 2,270 pre-augmented images in the archive were intentionally excluded to prevent data leakage into validation/test splits.
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print("\nDataset preparation completed successfully!")
        print(f"Saved into: {OUTPUT_DIR.resolve()}")
        print(f"Train: {len(b_train) + len(m_train)} images")
        print(f"Val:   {len(b_val) + len(m_val)} images")
        print(f"Test:  {len(b_test) + len(m_test)} images")
        print(f"Manifest saved at: {manifest_path}")

if __name__ == "__main__":
    prepare_binary_dataset()
