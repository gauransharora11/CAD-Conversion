import os

edge_dir = "edges"
cad_dir = "cad_targets"

edge_files = sorted(os.listdir(edge_dir))
cad_files = sorted(os.listdir(cad_dir))

print(f"Edges: {len(edge_files)} | CAD: {len(cad_files)}")

for edge, cad in zip(edge_files, cad_files):
    new_name = edge  # make CAD name same as edge
    os.rename(os.path.join(cad_dir, cad),
              os.path.join(cad_dir, new_name))

print("✅ CAD files renamed to match edges")
