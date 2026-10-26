"""
ORALNOVA-AI - Lightweight Deep Learning for Oral Lesion Screening
src package init
"""

__version__ = "1.0.0"
__author__ = "ORALNOVA-AI Research Prototype"

CLASS_NAMES = [
    "AphthousUlcer",
    "Gingivitis",
    "Leukoplakia",
    "Normal",
    "OralCancer",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "Normal": "Healthy oral tissue with no visible pathological changes.",
    "AphthousUlcer": "Canker sore - painful, shallow ulcers inside the mouth.",
    "Leukoplakia": "White patch/plaque on mucosa that cannot be wiped off - potential precancer.",
    "OralCancer": "Malignant lesion - requires urgent clinical evaluation.",
    "Gingivitis": "Inflammation of the gums caused by bacterial plaque.",
    "Benign": "Benign oral lesion - non-malignant tissue alteration.",
    "Malignant": "Malignant oral lesion - suspicious for malignancy, clinical referral indicated.",
}

CLASS_RISK = {
    "Normal": "low",
    "AphthousUlcer": "low",
    "Leukoplakia": "medium",
    "OralCancer": "high",
    "Gingivitis": "low",
    "Benign": "low",
    "Malignant": "high",
}

BINARY_CLASS_NAMES = [
    "Benign",
    "Malignant",
]
BINARY_NUM_CLASSES = len(BINARY_CLASS_NAMES)

