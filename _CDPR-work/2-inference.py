"""
Etape 2 : Inférence REFLECT sur des slices PNG préprocessées.

Charge le modèle REFLECT + le VAE Medical, fait tourner le Euler solver
(rectified flow) sur chaque slice, et sauvegarde les résultats intermédiaires
(encoded latents, latent corrigé, image reconstruite) dans un fichier .npz
par slice — utilisé ensuite par 3-anomaly-maps.py.

Usage :
    python _CDPR-work/2-inference.py \
        --model-path ./REFLECT_.../checkpoints/last.pt \
        --slices-dir /path/to/output-slices/test \
        --out-dir    /path/to/results \
        [--backward-steps 10] \
        [--batch-size 8]

Le config YAML du modèle (vae, image_size, etc.) est lu automatiquement
depuis le dossier parent du checkpoint, comme dans evaluate_REFLECT.py.
"""

from pathlib import Path
import sys
import argparse
import os
import yaml
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download

# ── Ajouter la racine du repo au path (ldm, medical_models, taming) ──────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in [PROJECT_ROOT, PROJECT_ROOT / "taming-transformers"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medical_models import UNET_models


# ─────────────────────────────────────────────────────────────────────────────
# Dataset minimal : lit les PNG déjà préprocessées
# ─────────────────────────────────────────────────────────────────────────────

class SliceDataset(Dataset):
    """Lit les PNG T1 depuis un dossier. Renvoie (tensor, stem)."""

    def __init__(self, slices_dir: str, image_size: int = 256):
        self.paths = sorted(glob(os.path.join(slices_dir, "*-T1.png")))
        if not self.paths:
            raise FileNotFoundError(f"Aucun fichier *-T1.png dans {slices_dir}")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("L")
        img = ImageOps.pad(img, (self.image_size, self.image_size), color=0)
        tensor = self.transform(np.array(img).astype(np.float32) / 255.0)
        stem = Path(path).stem.replace("-T1", "")
        return tensor, stem


# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────

def run_euler(model, encoded: torch.Tensor, backward_steps: int, device) -> torch.Tensor:
    """Euler solver du rectified flow (copié de evaluate_REFLECT.py)."""
    latent = encoded.clone()
    dt = 1.0 / backward_steps
    for time in torch.arange(0, 1, dt):
        t = time * torch.ones((latent.shape[0], 1), dtype=torch.float32, device=device)
        velocity = model(latent, t)
        latent = latent + velocity * dt
    return latent


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Lire la config YAML du checkpoint ────────────────────────────────────
    yaml_path = Path(args.model_path).parents[1] / "args.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"args.yml introuvable dans {yaml_path.parent}\n"
            "Spécifie manuellement --vae, --model, --image-size si nécessaire."
        )
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    vae_name   = config["vae"]          # "kl_f8" ou "kl_f4"
    model_name = config["model"]        # "UNet_M", etc.
    image_size = int(config["image_size"])

    embedding_dim       = 4 if vae_name == "kl_f8" else 3
    compression_factor  = 8 if vae_name == "kl_f8" else 4
    vae_filename        = "VAE-Medical-klf8.pt" if vae_name == "kl_f8" else "VAE-Medical-klf4.pt"

    print(f"Modèle : {model_name} | VAE : {vae_name} | Image size : {image_size}")

    # ── Charger le modèle REFLECT ─────────────────────────────────────────────
    model = UNET_models[model_name](in_channels=embedding_dim, out_channels=embedding_dim)
    state_dict = torch.load(args.model_path, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # ── Charger le VAE ────────────────────────────────────────────────────────
    vae_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename=vae_filename)
    vae = torch.load(vae_path, map_location="cpu", weights_only=False)
    vae.eval().to(device)

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    dataset = SliceDataset(args.slices_dir, image_size=image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, drop_last=False)
    print(f"{len(dataset)} slices à traiter.")

    out_dir = Path(args.out_dir) / "latents"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Boucle d'inférence ────────────────────────────────────────────────────
    for x_batch, stems in loader:
        x_batch = x_batch.to(device)

        with torch.no_grad():
            # Encode + normalisation (scale factor LDM standard)
            encoded = vae.encode(x_batch).mean.mul_(0.18215)

            # Rectified flow : latent corrigé (image "saine" reconstruite)
            latent_corrected = run_euler(model, encoded, args.backward_steps, device)

            # Decode dans l'espace image
            img_original     = vae.decode(encoded          / 0.18215)
            img_reconstructed = vae.decode(latent_corrected / 0.18215)

        # Sauvegarder par slice
        for i, stem in enumerate(stems):
            np.savez_compressed(
                out_dir / f"{stem}.npz",
                encoded           = encoded[i].cpu().numpy(),           # (C, h, w)
                latent_corrected  = latent_corrected[i].cpu().numpy(),  # (C, h, w)
                img_original      = img_original[i].cpu().numpy(),      # (1, H, W)
                img_reconstructed = img_reconstructed[i].cpu().numpy(), # (1, H, W)
            )

    print(f"Terminé. Résultats dans : {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence REFLECT sur slices PNG")
    parser.add_argument("--model-path",     required=True,
                        help="Chemin vers le checkpoint REFLECT (last.pt)")
    parser.add_argument("--slices-dir",     required=True,
                        help="Dossier contenant les PNG *-T1.png préprocessées")
    parser.add_argument("--out-dir",        required=True,
                        help="Dossier de sortie pour les .npz")
    parser.add_argument("--backward-steps", type=int, default=10,
                        help="Nombre de pas Euler (+ = précis, - = rapide, défaut 10)")
    parser.add_argument("--batch-size",     type=int, default=8)
    args = parser.parse_args()
    main(args)
