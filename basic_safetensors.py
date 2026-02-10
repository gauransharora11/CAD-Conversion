from safetensors import safe_open

tensors = {}

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(tensors)
