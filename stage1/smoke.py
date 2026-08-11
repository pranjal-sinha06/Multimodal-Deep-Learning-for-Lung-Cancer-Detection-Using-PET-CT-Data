import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("mem (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
