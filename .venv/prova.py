import torch

print("CUDA disponibile:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Dispositivo:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

