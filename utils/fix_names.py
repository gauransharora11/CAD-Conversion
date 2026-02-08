import os

EDGE_DIR = "edges"
CAD_DIR = "cad_targets"

edge_files = sorted(os.listdir(EDGE_DIR))
cad_files = sorted(os.listdir(CAD_DIR))

print("Renaming CAD files to match EDGE files...\n")

for i, (edge, cad) in enumerate(zip(edge_files, cad_files)):
    ext = os.path.splitext(edge)[1]  # keep same extension
    new_name = f"img_{i}.jpg"

    os.rename(os.path.join(EDGE_DIR, edge),
              os.path.join(EDGE_DIR, new_name))

    os.rename(os.path.join(CAD_DIR, cad),
              os.path.join(CAD_DIR, new_name))

    print(f"{edge}  <-->  {cad}  →  {new_name}")

print("\n✅ All files matched!")
