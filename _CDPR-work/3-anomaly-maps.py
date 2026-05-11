"""
Etape 3 : Génération des cartes d'anomalie et masques de lésion.

Lit les .npz produits par 2-inference.py et calcule pour chaque slice :
  - anomaly_map = 0.5 * diff_image + 0.5 * diff_latent  (méthode REFLECT)
  - masque binaire par seuillage d'Otsu (ou seuil fixe)

Sauvegarde :
  <out_dir>/
    anomaly_maps/   ← PNG float normalisé [0,255] (heatmap JET)
    masks/          ← PNG binaire uint8

Usage :
    python _CDPR-work/3-anomaly-maps.py \
        --npz-dir  /path/to/results/latents \
        --out-dir  /path/to/results \
        [--threshold otsu|<float 0-1>]
"""

from pathlib import Path
import argparse
import numpy as np
from glob import glob
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import cv2
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Carte d'anomalie (reproduit evaluate() de evaluate_REFLECT.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_anomaly_map(npz_path: Path, image_size: int = 256) -> np.ndarray:
    data = np.load(npz_path)

    img_orig  = data["img_original"]      # (1, H, W)  float, [-1,1]
    img_recon = data["img_reconstructed"] # (1, H, W)
    enc       = data["encoded"]           # (C, h, w)
    lat       = data["latent_corrected"]  # (C, h, w)

    # Différence image (clip + scale comme dans REFLECT)
    image_diff = np.abs(img_recon - img_orig).mean(axis=0)          # (H, W)
    image_diff = np.clip(image_diff, 0.0, 0.4) * 2.5
    image_diff = gaussian_filter(image_diff, sigma=3)

    # Différence latente (upscale vers image_size)
    latent_diff = np.abs(lat - enc).mean(axis=0)                     # (h, w)
    latent_diff = np.clip(latent_diff, 0.0, 0.4) * 2.5
    latent_diff = gaussian_filter(latent_diff, sigma=1)
    latent_diff = resize(latent_diff, (image_size, image_size), anti_aliasing=True)

    anomaly = 0.5 * image_diff + 0.5 * latent_diff
    return anomaly.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Seuillage → masque binaire
# ─────────────────────────────────────────────────────────────────────────────

def threshold_map(anomaly_map: np.ndarray, threshold) -> np.ndarray:
    """
    threshold : 'otsu' ou float [0,1].
    Retourne masque uint8 {0,255}.
    """
    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    img_u8 = (norm * 255).astype(np.uint8)

    if threshold == "otsu":
        _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = float(threshold)
        mask = (norm > th).astype(np.uint8) * 255

    # Nettoyage morphologique : supprime les petites composantes isolées
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    npz_dir  = Path(args.npz_dir)
    out_dir  = Path(args.out_dir)
    map_dir  = out_dir / "anomaly_maps"
    mask_dir = out_dir / "masks"
    map_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"Aucun .npz dans {npz_dir}")

    print(f"{len(npz_files)} slices à traiter.")

    for npz_path in tqdm(npz_files, desc="Anomaly maps"):
        stem = npz_path.stem

        anomaly = compute_anomaly_map(npz_path, image_size=args.image_size)

        # ── Heatmap couleur JET ───────────────────────────────────────────────
        norm_u8 = ((anomaly - anomaly.min()) /
                   (anomaly.max() - anomaly.min() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR→RGB
        Image.fromarray(heatmap).save(map_dir / f"{stem}-anomaly.png")

        # ── Masque binaire ────────────────────────────────────────────────────
        mask = threshold_map(anomaly, args.threshold)
        Image.fromarray(mask).save(mask_dir / f"{stem}-mask.png")

        # ── Sauvegarde float brut (pour back-to-nifti) ───────────────────────
        np.save(map_dir / f"{stem}-anomaly.npy", anomaly)

    print(f"Terminé.")
    print(f"  Heatmaps PNG : {map_dir.resolve()}")
    print(f"  Masques PNG  : {mask_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération cartes d'anomalie REFLECT")
    parser.add_argument("--npz-dir",    required=True,
                        help="Dossier contenant les .npz de 2-inference.py (sous-dossier latents/)")
    parser.add_argument("--out-dir",    required=True,
                        help="Dossier de sortie")
    parser.add_argument("--threshold",  default="otsu",
                        help="'otsu' (défaut) ou float [0-1] ex: 0.5")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    main(args)
