from pathlib import Path
import sys

from huggingface_hub import hf_hub_download
import torch


# Ensure repo root is importable (required to unpickle modules like `ldm.*`)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# taming-transformers cloned locally (pip package is broken)
TAMING_ROOT = PROJECT_ROOT / "taming-transformers"
if TAMING_ROOT.exists() and str(TAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(TAMING_ROOT))

# Download model
model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
print(model_path)


# # Load the model
model = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)

model.eval()


def _infer_in_channels(m):
    if hasattr(m, "encoder") and hasattr(m.encoder, "conv_in") and hasattr(m.encoder.conv_in, "in_channels"):
        return int(m.encoder.conv_in.in_channels)
    return 1


def _extract_latent(encoded):
    # AutoencoderKL path
    if hasattr(encoded, "sample"):
        return encoded.sample()
    # VQ path
    if isinstance(encoded, (tuple, list)) and len(encoded) > 0:
        return encoded[0]
    return encoded


in_channels = _infer_in_channels(model)
x = torch.randn(1, in_channels, 256, 256)

with torch.no_grad():
    encoded = model.encode(x)
    latent = _extract_latent(encoded)
    reconstruction = model.decode(latent)

print(f"Model type: {type(model)}")
print(f"Input shape: {tuple(x.shape)}")
print(f"Latent shape: {tuple(latent.shape)}")
print(f"Reconstruction shape: {tuple(reconstruction.shape)}")

print("Load + encode/decode test: OK")
