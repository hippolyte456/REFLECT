"""
Preprocessing de T1w personnels → PNG slices compatibles REFLECT.

Pipeline :
    1. Registration affine vers MNI152 1mm (ANTs, full-head → full-head)
    2. Application du masque cerveau MNI → skull stripping (pas de FSL/antspynet)
    3. Normalisation d'intensité robuste (percentile 1–99% dans le masque)
    4. Slicing axial + padding 256×256 (identique au DataLoader REFLECT)

Input : fichier NIfTI unique (--input-file) ou dossier (--input-dir).
        Optionnel : masques de lésion au format {stem}_lesion.nii.gz (pour évaluation).

Output :
    <out_dir>/
        test/
            {subject_id}-slice_{z:03d}-T1.png
            {subject_id}-slice_{z:03d}-brainmask.png
            {subject_id}-slice_{z:03d}-segmentation.png  ← si masque lésion fourni

Usage :
    python _CDPR-work/1-preprocess-custom.py \
        --input-file /path/to/T1w.nii.gz \
        --out-dir    /path/to/output \
        [--split train|test]

Dépendances :
    pip install antspyx nibabel tqdm
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATES_DIR    = Path(__file__).resolve().parent / "templates"
DEFAULT_MNI      = str(TEMPLATES_DIR / "MNI152_T1_1mm.nii.gz")
DEFAULT_MNI_MASK = str(TEMPLATES_DIR / "MNI152_T1_1mm_brain_mask.nii.gz")
IMAGE_SIZE = 256
MIN_BRAIN_RATIO = 0.02   # fraction minimale de voxels cerveau pour garder une coupe


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 1 : Registration MNI (ANTs)
# ─────────────────────────────────────────────────────────────────────────────

def register_to_mni(t1_path: Path, mni_template: str, out_path: Path) -> Path:
    """Registration affine ANTs vers MNI152 1mm."""
    import ants
    print(f"  [ANTs] Registration → MNI : {t1_path.name}")
    fixed  = ants.image_read(mni_template)
    moving = ants.image_read(str(t1_path))
    result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")
    ants.image_write(result["warpedmovout"], str(out_path))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 : Skull stripping via masque MNI (pas de FSL / antspynet requis)
# ─────────────────────────────────────────────────────────────────────────────

def apply_mni_mask(registered_path: Path, mni_mask: str,
                   out_brain: Path, out_mask: Path) -> tuple[Path, Path]:
    """
    Applique le masque cerveau MNI152 sur le T1 déjà recalé.
    Pas besoin de FSL BET ni de antspynet.
    """
    print(f"  [MNI mask] Skull stripping par masque MNI")
    brain_img = nib.load(str(registered_path))
    mask_img  = nib.load(mni_mask)
    brain_data = brain_img.get_fdata(dtype=np.float32)
    mask_data  = (mask_img.get_fdata() > 0.5).astype(np.uint8)
    nib.save(nib.Nifti1Image(brain_data * mask_data, brain_img.affine), str(out_brain))
    nib.save(nib.Nifti1Image(mask_data,               brain_img.affine), str(out_mask))
    return out_brain, out_mask


# ─────────────────────────────────────────────────────────────────────────────
# ETAPES 3–4 : Normalisation + slicing + sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical(path: Path) -> tuple[np.ndarray, object]:
    """Charge un NIfTI et réoriente en RAS (axial = axis 2)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32), img.affine


def normalize_intensity(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Normalisation robuste percentile 1–99% dans le masque cerveau → uint8."""
    brain_vox = volume[brain_mask > 0]
    if len(brain_vox) == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    p1, p99 = np.percentile(brain_vox, 1), np.percentile(brain_vox, 99)
    norm = np.clip((volume - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    norm[brain_mask == 0] = 0.0
    return (norm * 255).astype(np.uint8)


def to_pil_padded(slice_2d: np.ndarray) -> Image.Image:
    """Slice 2D → PIL Image paddée à IMAGE_SIZE×IMAGE_SIZE (même que DataLoader REFLECT)."""
    img = Image.fromarray(slice_2d.astype(np.uint8))
    return ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=0)


def save_slices(subject_id: str,
                t1_vol: np.ndarray,
                brain_mask_vol: np.ndarray,
                lesion_vol: np.ndarray | None,
                out_dir: Path,
                split: str) -> int:
    """Slice axial + save PNG. Retourne le nombre de coupes sauvegardées."""
    t1_norm = normalize_intensity(t1_vol, brain_mask_vol)
    n_slices = t1_vol.shape[2]
    saved = 0

    for z in range(n_slices):
        brain_slice = brain_mask_vol[:, :, z]
        ratio = brain_slice.sum() / (brain_slice.shape[0] * brain_slice.shape[1])
        if ratio < MIN_BRAIN_RATIO:
            continue

        prefix = str(out_dir / f"{subject_id}-slice_{z:03d}")

        to_pil_padded(t1_norm[:, :, z]).convert("L").save(f"{prefix}-T1.png")
        to_pil_padded((brain_slice * 255).astype(np.uint8)).convert("L").save(f"{prefix}-brainmask.png")

        if lesion_vol is not None and split == "test":
            lesion_slice = (lesion_vol[:, :, z] > 0.5).astype(np.uint8) * 255
            to_pil_padded(lesion_slice).convert("L").save(f"{prefix}-segmentation.png")

        saved += 1

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(t1_path: Path,
                    lesion_path: Path | None,
                    out_dir: Path,
                    split: str,
                    mni_template: str,
                    mni_mask: str,
                    tmp_dir: Path) -> int:

    subject_id = t1_path.stem.replace(".nii", "")  # gère .nii et .nii.gz
    subj_tmp = tmp_dir / subject_id
    subj_tmp.mkdir(parents=True, exist_ok=True)

    registered_path = subj_tmp / "T1_MNI.nii.gz"
    brain_path      = subj_tmp / "T1_brain.nii.gz"
    mask_path       = subj_tmp / "brain_mask.nii.gz"

    # 1. Registration full-head → MNI full-head
    if not registered_path.exists():
        register_to_mni(t1_path, mni_template, registered_path)

    # 2. Skull stripping par masque MNI
    if not brain_path.exists() or not mask_path.exists():
        apply_mni_mask(registered_path, mni_mask, brain_path, mask_path)

    # 3. Load & normalize
    t1_vol,         _ = load_canonical(brain_path)
    brain_mask_vol, _ = load_canonical(mask_path)
    brain_mask_vol     = (brain_mask_vol > 0.5).astype(np.uint8)

    lesion_vol = None
    if lesion_path and lesion_path.exists():
        lesion_vol, _ = load_canonical(lesion_path)

    # 4. Slice & save
    return save_slices(subject_id, t1_vol, brain_mask_vol, lesion_vol, out_dir, split)


def main(args):
    out_dir = Path(args.out_dir) / args.split
    tmp_dir = Path(args.out_dir) / "_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Fichier unique ou dossier
    if args.input_file:
        t1_files = [args.input_file]
    else:
        input_dir = Path(args.input_dir)
        t1_files = sorted(glob(str(input_dir / "*.nii.gz")) + glob(str(input_dir / "*.nii")))
        if not t1_files:
            raise FileNotFoundError(f"Aucun fichier NIfTI trouvé dans {input_dir}")

    print(f"Trouvé {len(t1_files)} fichier(s) T1w.")

    total = 0
    for t1_path in tqdm(t1_files, desc="Sujets"):
        t1_path = Path(t1_path)

        # Chercher masque de lésion optionnel : {stem}_lesion.nii.gz
        stem = t1_path.name.replace(".nii.gz", "").replace(".nii", "")
        lesion_candidates = list(t1_path.parent.glob(f"{stem}*lesion*.nii*"))
        lesion_path = lesion_candidates[0] if lesion_candidates else None

        try:
            n = process_subject(
                t1_path=t1_path,
                lesion_path=lesion_path,
                out_dir=out_dir,
                split=args.split,
                mni_template=args.mni_template,
                mni_mask=args.mni_mask,
                tmp_dir=tmp_dir,
            )
            total += n
            print(f"  → {n} coupes sauvegardées")
        except Exception as e:
            print(f"  ✗ Erreur sur {t1_path.name} : {e}")

    print(f"\nTerminé. {total} coupes PNG dans : {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Préprocess T1w custom → PNG slices REFLECT-compatibles"
    )
    parser.add_argument("--input-file", default=None,
                        help="Fichier NIfTI T1w unique (.nii ou .nii.gz)")
    parser.add_argument("--input-dir", default=None,
                        help="Dossier contenant les NIfTI T1w (.nii ou .nii.gz)")
    parser.add_argument("--out-dir", required=True,
                        help="Dossier de sortie (les slices iront dans out-dir/train ou test/)")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="'test' pour inférence (défaut), 'train' pour données saines")
    parser.add_argument("--mni-template", default=DEFAULT_MNI,
                        help="Template MNI152 full-head 1mm (défaut: templates/ du repo)")
    parser.add_argument("--mni-mask", default=DEFAULT_MNI_MASK,
                        help="Masque cerveau MNI152 1mm (défaut: templates/ du repo)")
    args = parser.parse_args()
    main(args)
