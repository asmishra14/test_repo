import os

folder = "IPL_2026-GT-PBKS"

for i, filename in enumerate(os.listdir(folder), start=1):
    ext = os.path.splitext(filename)[1]  # keep original extension
    new_name = f"IPL-2026-GT-PBKS-{i}-KSK{ext}"
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print("Renamed:", filename, "→", new_name)
