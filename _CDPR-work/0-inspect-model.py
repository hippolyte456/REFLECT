import torch
from huggingface_hub import hf_hub_download

# Télécharger le modèle
model_path = hf_hub_download(
    repo_id="farzadbz/Medical-VAE",
    filename="VAE-Medical-klf8.pt"
)

print(f"Model path: {model_path}")

# Charger sur CPU
model = torch.load(model_path, map_location="cpu")

print("\n=== MODEL TYPE ===")
print(type(model))

print("\n=== MODEL ARCHITECTURE ===")
print(model)

print("\n=== MODEL METHODS ===")
methods = [m for m in dir(model) if not m.startswith("_")]
print(methods)

print("\n=== PARAMETER COUNT ===")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params     : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")

print("\n=== MODEL DEVICE ===")
print(next(model.parameters()).device)

print("\n=== MODEL IN EVAL MODE ===")
model.eval()
print(model.training)

print("\n=== FIRST PARAMETER SHAPE ===")
first_param = next(model.parameters())
print(first_param.shape)

print("\n=== ENCODE / DECODE AVAILABLE ===")
print("encode :", hasattr(model, "encode"))
print("decode :", hasattr(model, "decode"))

# Test d'un faux input
print("\n=== DUMMY FORWARD TEST ===")

try:
    # Adapter la taille si nécessaire
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        encoded = model.encode(x)

    print("Encoding success")
    print(type(encoded))

except Exception as e:
    print("Error during test:")
    print(e)