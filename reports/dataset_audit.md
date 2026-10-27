# ORALNOVA-AI: Dataset Audit & Feasibility Report

**Project**: ORALNOVA-AI (Lightweight Oral Lesion Screening Prototype)  
**Date of Audit**: October 27, 2026  
**Auditor**: Automated Research Integrity Audit  
**Status**: Preliminary Audit Complete — Training Halted (No fabrication permitted)

---

## 1. Executive Summary

This audit evaluated the oral imaging archive currently obtained on the system (`C:\Users\DELL\Downloads\mhjyrn35p4-2.zip`, Mendeley Data Version 2), cross-referenced with the referenced **CODE** (*Comprehensive Oral Mucosa Database with Explanations*) dataset.

### Key Audit Findings:
1. **Critical Label Mismatch**:
   - The archive supports strictly **2 binary classes**: `Benign Lesion` (165 images) and `Malignant Lesion` (158 images).
   - It **does NOT contain** your configured 5 classes (`AphthousUlcer`, `Gingivitis`, `Leukoplakia`, `Normal`, `OralCancer`).
   - Specifically, healthy/normal oral tissue, gingivitis, and aphthous ulcers are not labeled in this dataset.
2. **Data Integrity & Pre-Augmentation**:
   - Total images in archive: **2,593 images**.
   - Original un-augmented images: **323 images** (165 benign, 158 malignant).
   - Pre-augmented images: **2,270 images** (~7x rotated/flipped copies created by the original authors).
   - **Crucial Rule**: Pre-augmented images **must never be used in validation or test splits**. Testing on augmented copies of training images causes severe data leakage and artificially inflated metrics.
3. **Absence of Patient Identifiers**:
   - Filenames are sequential integers (`1.jpg`, `2.jpg`, etc.).
   - EXIF metadata contains camera hardware tags (`Samsung SM-G360H`, `Windows Photo Editor`), but **zero patient IDs, hospital IDs, or clinical demographics**.
   - True patient-level splitting is **not possible** with this archive alone. An image-level stratified split on original data is the only reproducible approach.
4. **Licence & Usage Rights**:
   - **Licence**: Creative Commons Attribution 4.0 International (**CC BY 4.0**).
   - Permitted: Academic research, modification, and prototyping, provided appropriate attribution is given to the authors (Chandrashekar et al., 2021).
   - Medical Disclaimer: Academic research prototype only; not cleared for primary clinical diagnosis.

---

## 2. Archive File Structure & Inventory

### Archive Details
- **Source File**: `C:\Users\DELL\Downloads\mhjyrn35p4-2.zip`
- **DOI**: `10.17632/mhjyrn35p4.2` (Mendeley Data)
- **Title**: *Oral Images Dataset*
- **Authors**: Chandrashekar H. S., Geetha Kiran A., Murali S., Dinesh M. S., Nanditha B. R. (2021)
- **Licence**: CC BY 4.0

### Directory Structure & File Counts

```text
Oral Images Dataset/
├── original_data/
│   ├── benign_lesions/        --> 165 JPG images
│   └── malignant_lesions/     --> 158 JPG images
└── augmented_data/
    ├── augmented_benign/      --> 1,155 JPG images (pre-augmented)
    └── augmented_malignant/   --> 1,115 JPG images (pre-augmented)
```

| Partition | Original Benign | Original Malignant | Total Original | Augmented Copies | Total Archive Files |
|---|---|---|---|---|---|
| **Counts** | 165 | 158 | **323** | 2,270 | **2,593** |

- **Annotation / Metadata Files**: None present in the archive (no CSV, XML, or JSON metadata).
- **Corrupt Images**: **0** (all 2,593 images opened and verified successfully using PIL).
- **Bit Depth & Format**: 100% 24-bit RGB JPEG images.
- **Resolutions**: 323 unique image dimensions (ranging from 205x136 to 2560x1536), confirming that the augmented images share the exact dimensions of the 323 original images.

---

## 3. Label Compatibility Analysis

| Configured ORALNOVA-AI Class | Present in `mhjyrn35p4-2.zip`? | Status / Comment |
|---|---|---|
| `Normal` (Healthy tissue) | **NO** | Missing. All images depict pathological lesions. |
| `AphthousUlcer` | **NO** | Not separately labeled (subsumed under general benign). |
| `Gingivitis` | **NO** | Missing. |
| `Leukoplakia` | **NO** | Not separately labeled (subsumed under benign / precancer). |
| `OralCancer` | **YES** | Matches `malignant_lesions` (158 original images). |

### What About the Official CODE Dataset?
The **CODE** (*Comprehensive Oral Mucosa Database with Explanations*) dataset developed under the ICMR-SMITA framework on Figshare contains:
1. `Normal Mucosa`
2. `Variations from Normal`
3. `Oral Potentially Malignant Disorders (OPMD)`
4. `Oral Cancer`

Neither `mhjyrn35p4` nor the official `CODE` dataset includes a standalone 5-class breakdown for `AphthousUlcer` and `Gingivitis`. Attempting to train a 5-class classifier on binary or 4-class data would require inventing labels, which is unacceptable and clinically invalid.

---

## 4. Patient Leakage & Splitting Assessment

### Patient Leakage Audit:
- **Available Identifiers**: None.
- **Risk**: If multiple images were acquired from the same subject (e.g. multiple camera angles or follow-up visits), an image-level split risks placing images of the same patient in both `train/` and `test/`.
- **Mitigation for Academic Demonstration**:
  1. Acknowledge this limitation explicitly in your project report/presentation.
  2. Use **only the 323 original images** for dataset partitioning.
  3. Never place images from `augmented_data/` into validation or test sets. Any data augmentation must be performed on-the-fly inside PyTorch's `train_loader` only.

### Recommended Clean Partitioning (323 Original Images):
- **Train (70%)**: ~226 images (115 benign, 111 malignant)
- **Validation (15%)**: ~48 images (25 benign, 23 malignant)
- **Test (15%)**: ~49 images (25 benign, 24 malignant)

---

## 5. Recommended Next Steps

### Option A: Binary Screening Task (Recommended for Demonstration)
Adapt the application head to the genuine binary task supported by this dataset:
1. **Target Classes**: `["Benign", "Malignant"]`.
2. **Clinical Role**: Early referral screening (Suspicious Malignant vs. Benign Oral Lesion).
3. **Model Changes**: Minimal change — `NUM_CLASSES = 2`. The rest of MobileNetV3-Small, Grad-CAM, image quality checks, and Streamlit UI remain identical.

### Option B: Wait for Multi-Class CODE / MeMoSA Dataset
If your college project strictly requires the 5 pre-configured classes:
1. Obtain an institutional or multi-class dataset that explicitly annotates `AphthousUlcer`, `Gingivitis`, `Leukoplakia`, `Normal`, and `OralCancer`.
2. Do not train until those exact folders are populated with validated clinical ground-truth images.

---

## 6. Audit Verdict

- **Fabrication**: Rejected. No fake labels or synthetic checkpoints were created.
- **Safety**: Passed. All project code, unit tests, and dashboard assets remain intact.
- **Decision Required**: Confirm whether to proceed with **Option A (Binary Lesion Screening: Benign vs Malignant)** using the 323 original images.

## 7. Day 1–2 Verification Update

### Dataset Inventory
The binary dataset contains 323 images:
- Train: 225 images (115 Benign, 110 Malignant)
- Validation: 49 images (25 Benign, 24 Malignant)
- Test: 49 images (25 Benign, 24 Malignant)

### Integrity Checks Performed
- Manifest entries: 323.
- Missing or mismatched files during SHA-256 manifest verification: 0.
- Exact duplicate hash groups in the manifest: 0.
- Actual image files independently hashed: 323.
- Exact duplicate groups in the independent image scan: 0.
- Manifest split-path mismatches: 0.

### Limitations
These checks establish file integrity and split-path consistency only. They do not rule out patient-level leakage, visually similar images, or clinical limitations. Patient identifiers were unavailable in the source dataset.

### Progress
- Day 1: Dataset inventory and class compatibility audit documented.
- Day 2: File hashes, exact duplicates, and split assignments checked.
- Day 3: Test-set evaluation remains pending.

## 7. Day 1-2 Verification Update

### Dataset Inventory
- Train: 225 images (115 Benign, 110 Malignant).
- Validation: 49 images (25 Benign, 24 Malignant).
- Test: 49 images (25 Benign, 24 Malignant).
- Total: 323 original images.

### Integrity Checks
- Manifest entries: 323.
- Missing or mismatched files during SHA-256 verification: 0.
- Exact duplicate hash groups in manifest: 0.
- Independently hashed image files: 323.
- Exact duplicate groups in independent scan: 0.
- Manifest split-path mismatches: 0.

### Limitations
- Patient identifiers were unavailable; patient-level leakage cannot be ruled out.
- Exact hash checks do not detect visually similar images or images from the same patient.
- These checks do not establish clinical validity or model performance.

### Progress
- Day 1: Dataset inventory and class compatibility audit documented.
- Day 2: File integrity, exact duplicates, and split-path consistency checked.
- Day 3: Test-set evaluation remains pending.
