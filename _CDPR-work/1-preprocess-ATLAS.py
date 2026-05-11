"""
Preprocessing ATLAS v2.1 T1w MRI → REFLECT-compatible PNG slices.

Expected input structure (BIDS derivatives, fully-normalized version):
    <atlas_root>/
        sub-r001s001/
            ses-1/
                anat/
                    sub-r001s001_ses-1_T1w.nii.gz
        derivatives/
            sub-r001s001/
                ses-1/
                    sub-r001s001_ses-1_label-L_desc-T1lesion_mask.nii.gz

Output structure:
    <out_root>/
        train/
            sub-r001s001-slice_072-T1.png
            sub-r001s001-slice_072-brainmask.png
        test/
            sub-r001s001-slice_072-T1.png
            sub-r001s001-slice_072-brainmask.png
            sub-r001s001-slice_072-segmentation.png

Notes:
- Uses ATLAS v2.1 "fully normalized" (already MNI152 1mm, skull-stripped).
- If your data is NOT already in MNI space, set REGISTER=True (requires ANTs).
- Brain mask = any voxel with intensity > 0 after skull-strip.
- Intensity normalization: robust percentile (1–99%) within brain, → uint8.
- Axial slicing: only slices with ≥ MIN_BRAIN_RATIO brain coverage kept.
- Padding: PIL ImageOps.pad → 256×256, black background (identical to REFLECT DataLoader).
"""

import argparse
import os
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REGISTER = False          # True if data is NOT already in MNI152 space
MNI_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
IMAGE_SIZE = 256
MIN_BRAIN_RATIO = 0.02    # skip slices with < 2% brain voxels
TRAIN_RATIO = 0.80        # 80% train / 20% test split (by subject)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def register_to_mni(t1_path: Path, out_path: Path) -> Path:
    """Register T1w to MNI152 using ANTs (only needed if REGISTER=True)."""
    try:
        import ants
    except ImportError:
        raise ImportError("antspy is required for registration: pip install antspyx")

    fixed = ants.image_read(MNI_TEMPLATE)
    moving = ants.image_read(str(t1_path))
    result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")
    warped = result["warpedmovout"]
    ants.image_write(warped, str(out_path))
    return out_path


def load_nifti_as_numpy(path: Path) -> tuple[np.ndarray, object]:
    """Load NIfTI, reorient to RAS (axial = last axis), return (data, affine)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)   # → RAS orientation
    return img.get_fdata(dtype=np.float32), img.affine


def normalize_intensity(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Robust percentile normalization within brain mask → uint8 [0, 255].
    Background (outside brain) stays 0.
    """
    brain_voxels = volume[brain_mask > 0]
    p1, p99 = np.percentile(brain_voxels, 1), np.percentile(brain_voxels, 99)
    norm = np.clip((volume - p1) / (p99 - p1 + 1e-8), 0, 1)
    norm[brain_mask == 0] = 0
    return (norm * 255).astype(np.uint8)


def slice_and_pad(volume_2d: np.ndarray) -> Image.Image:
    """
    Convert a 2D axial slice (H×W) to a PIL Image, pad to IMAGE_SIZE×IMAGE_SIZE
    with black background (same as REFLECT DataLoader's ImageOps.pad call).
    """
    img = Image.fromarray(volume_2d)
    img = ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=0)
    return img


def save_slice(img: Image.Image, path: Path):
    img.convert("L").save(str(path))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def find_atlas_subjects(atlas_root: Path):
    """
    Returns list of dicts with keys: subject_id, t1_path, lesion_path.
    Supports both ATLAS v2.1 BIDS structure and flat structure.
    """
    subjects = []

    # BIDS: sub-*/ses-*/anat/*_T1w.nii.gz
    t1_files = sorted(glob(str(atlas_root / "sub-*" / "ses-*" / "anat" / "*_T1w.nii.gz")))
    if not t1_files:
        # flat fallback: sub-*/*_T1w.nii.gz
        t1_files = sorted(glob(str(atlas_root / "sub-*" / "*_T1w.nii.gz")))

    for t1_path in t1_files:
        t1_path = Path(t1_path)
        subject_id = t1_path.parts[-4] if "ses-" in str(t1_path) else t1_path.parts[-3]

        # Find corresponding lesion mask (BIDS derivatives)
        stem = t1_path.name.replace("_T1w.nii.gz", "")
        lesion_candidates = sorted(glob(
            str(atlas_root / "**" / f"{stem}_label-L_desc-T1lesion_mask.nii.gz"),
            recursive=True
        ))
        if not lesion_candidates:
            # Try alternative naming
            lesion_candidates = sorted(glob(
                str(atlas_root / "**" / f"{stem}*lesion*.nii.gz"),
                recursive=True
            ))

        lesion_path = Path(lesion_candidates[0]) if lesion_candidates else None

        subjects.append({
            "subject_id": subject_id,
            "t1_path": t1_path,
            "lesion_path": lesion_path,
        })

    return subjects


def process_subject(subj: dict, out_root: Path, split: str, tmp_dir: Path):
    subject_id = subj["subject_id"]
    t1_path = subj["t1_path"]
    lesion_path = subj["lesion_path"]

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Optional MNI registration ──────────────────────────────────────
    if REGISTER:
        registered_path = tmp_dir / f"{subject_id}_T1w_MNI.nii.gz"
        if not registered_path.exists():
            t1_path = register_to_mni(t1_path, registered_path)
        else:
            t1_path = registered_path

    # ── 2. Load volumes ───────────────────────────────────────────────────
    t1_vol, _ = load_nifti_as_numpy(t1_path)

    lesion_vol = None
    if lesion_path and lesion_path.exists():
        lesion_vol, _ = load_nifti_as_numpy(lesion_path)

    # ── 3. Brain mask = nonzero T1 (already skull-stripped in ATLAS normalized) ──
    brain_mask = (t1_vol > 0).astype(np.uint8)

    # ── 4. Intensity normalization ─────────────────────────────────────────
    t1_norm = normalize_intensity(t1_vol, brain_mask)

    # ── 5. Axial slicing (z = last axis in RAS after reorientation) ────────
    n_slices = t1_vol.shape[2]
    saved = 0

    for z in range(n_slices):
        brain_slice = brain_mask[:, :, z]

        # Skip slices with too little brain
        brain_ratio = brain_slice.sum() / (brain_slice.shape[0] * brain_slice.shape[1])
        if brain_ratio < MIN_BRAIN_RATIO:
            continue

        t1_slice = t1_norm[:, :, z]
        mask_slice = (brain_slice * 255).astype(np.uint8)

        prefix = out_dir / f"{subject_id}-slice_{z:03d}"

        save_slice(slice_and_pad(t1_slice), Path(str(prefix) + "-T1.png"))
        save_slice(slice_and_pad(mask_slice), Path(str(prefix) + "-brainmask.png"))

        if split == "test" and lesion_vol is not None:
            lesion_slice = (lesion_vol[:, :, z] > 0.5).astype(np.uint8) * 255
            save_slice(slice_and_pad(lesion_slice), Path(str(prefix) + "-segmentation.png"))

        saved += 1

    return saved


def main(args):
    atlas_root = Path(args.atlas_dir)
    out_root = Path(args.out_dir)
    tmp_dir = out_root / "_tmp_registered"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_atlas_subjects(atlas_root)
    if not subjects:
        raise FileNotFoundError(f"No subjects found in {atlas_root}. Check directory structure.")

    print(f"Found {len(subjects)} subjects.")

    # Train/test split by subject (deterministic)
    n_train = int(len(subjects) * TRAIN_RATIO)
    train_subjects = subjects[:n_train]
    test_subjects = subjects[n_train:]

    print(f"Train: {len(train_subjects)} subjects | Test: {len(test_subjects)} subjects")

    total_slices = 0
    for subj in tqdm(train_subjects, desc="Train subjects"):
        n = process_subject(subj, out_root, "train", tmp_dir)
        total_slices += n

    for subj in tqdm(test_subjects, desc="Test subjects"):
        n = process_subject(subj, out_root, "test", tmp_dir)
        total_slices += n

    print(f"\nDone. Total slices saved: {total_slices}")
    print(f"Output: {out_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ATLAS v2.1 for REFLECT")
    parser.add_argument("--atlas-dir", required=True,
                        help="Root directory of ATLAS v2.1 dataset")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for PNG slices")
    args = parser.parse_args()
    main(args)
