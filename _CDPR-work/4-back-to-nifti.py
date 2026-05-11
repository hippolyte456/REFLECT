"""
Etape 4 : Réassemblage des masques 2D → volume NIfTI 3D.

Lit les masques PNG produits par 3-anomaly-maps.py, les réassemble
slice par slice dans l'ordre axial, et sauvegarde un .nii.gz par sujet
dans le même espace que le volume MNI préprocessé (récupère l'affine depuis
le NIfTI intermédiaire produit par 1-preprocess-custom.py).

Usage :
    python _CDPR-work/4-back-to-nifti.py \
        --masks-dir    /path/to/results/masks \
        --preproc-tmp  /path/to/output/_tmp \
        --out-dir      /path/to/results/nifti \
        [--image-size 256]

Structure attendue des PNG : {subject_id}-slice_{z:03d}-mask.png
"""

from pathlib import Path
import argparse
import re
from collections import defaultdict

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Parsing du naming
# ─────────────────────────────────────────────────────────────────────────────

SLICE_RE = re.compile(r"^(.+)-slice_(\d+)-mask$")


def parse_mask_files(masks_dir: Path) -> dict[str, dict[int, Path]]:
    """Retourne {subject_id: {z: mask_path}}."""
    subjects = defaultdict(dict)
    for p in sorted(masks_dir.glob("*-mask.png")):
        m = SLICE_RE.match(p.stem)
        if not m:
            continue
        subject_id = m.group(1)
        z = int(m.group(2))
        subjects[subject_id][z] = p
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# Récupérer l'affine MNI depuis le volume intermédiaire préprocessé
# ─────────────────────────────────────────────────────────────────────────────

def get_reference_affine_and_shape(subject_id: str, preproc_tmp: Path):
    """
    Cherche le NIfTI brain/MNI produit par 1-preprocess-custom.py dans _tmp/.
    Retourne (affine, (nx, ny, nz)).
    """
    candidates = list((preproc_tmp / subject_id).glob("T1_brain.nii.gz"))
    if not candidates:
        candidates = list((preproc_tmp / subject_id).glob("T1_MNI.nii.gz"))
    if not candidates:
        # Fallback : affine identité 1mm isotropique
        print(f"  ⚠  Aucun NIfTI de référence trouvé pour {subject_id}, affine identité utilisée.")
        return np.eye(4), None

    ref_img = nib.load(str(candidates[0]))
    ref_img = nib.as_closest_canonical(ref_img)
    return ref_img.affine, ref_img.shape[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Réassemblage
# ─────────────────────────────────────────────────────────────────────────────

def assemble_volume(slice_map: dict[int, Path],
                    ref_shape: tuple | None,
                    image_size: int) -> np.ndarray:
    """
    Réassemble les masques PNG en volume 3D (nx, ny, nz).
    Si ref_shape est connue, le volume final a les bonnes dimensions.
    Sinon on utilise le z max des slices disponibles.
    """
    z_indices = sorted(slice_map.keys())
    z_max = (ref_shape[2] if ref_shape else z_indices[-1] + 1)

    # Taille spatiale dans le plan : on relit la première slice pour la savoir
    first_mask = np.array(Image.open(slice_map[z_indices[0]]).convert("L"))
    ny, nx = first_mask.shape

    volume = np.zeros((nx, ny, z_max), dtype=np.uint8)

    for z, path in slice_map.items():
        if z >= z_max:
            continue
        mask_slice = np.array(Image.open(path).convert("L"))
        mask_slice = (mask_slice > 128).astype(np.uint8)
        volume[:, :, z] = mask_slice.T  # PIL = (W,H), NIfTI = (X,Y,Z)

    return volume


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    masks_dir   = Path(args.masks_dir)
    preproc_tmp = Path(args.preproc_tmp)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_mask_files(masks_dir)
    if not subjects:
        raise FileNotFoundError(f"Aucun masque *-mask.png trouvé dans {masks_dir}")

    print(f"{len(subjects)} sujet(s) à réassembler.")

    for subject_id, slice_map in tqdm(subjects.items(), desc="NIfTI"):
        affine, ref_shape = get_reference_affine_and_shape(subject_id, preproc_tmp)
        volume = assemble_volume(slice_map, ref_shape, args.image_size)

        nifti = nib.Nifti1Image(volume.astype(np.uint8), affine)
        out_path = out_dir / f"{subject_id}_lesion_mask.nii.gz"
        nib.save(nifti, str(out_path))

    print(f"Terminé. NIfTI dans : {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Réassemble les masques PNG → NIfTI 3D"
    )
    parser.add_argument("--masks-dir",    required=True,
                        help="Dossier contenant les *-mask.png (produit par 3-anomaly-maps.py)")
    parser.add_argument("--preproc-tmp",  required=True,
                        help="Dossier _tmp de 1-preprocess-custom.py (pour récupérer l'affine)")
    parser.add_argument("--out-dir",      required=True,
                        help="Dossier de sortie pour les .nii.gz")
    parser.add_argument("--image-size",   type=int, default=256)
    args = parser.parse_args()
    main(args)
