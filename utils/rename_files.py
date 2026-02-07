import os

for folder in ["photos", "cad_targets"]:
    if not os.path.exists(folder):
        continue

    for i, file in enumerate(os.listdir(folder)):
        ext = file.split(".")[-1]
        new_name = f"img_{i}.{ext}"
        os.rename(f"{folder}/{file}", f"{folder}/{new_name}")

print("Files renamed cleanly")