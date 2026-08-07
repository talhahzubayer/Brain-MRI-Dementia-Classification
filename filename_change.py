import os
from pathlib import Path

# Point this to your local dataset
DATA_DIR = Path("C:\\Users\\Talhah Zubayer\\Downloads\\AI Coursework\\AD")

class_folders = ["NoImpairment", "MildImpairment", "ModerateImpairment"]

# Dry run first - set to False to actually rename
DRY_RUN = False

for class_name in class_folders:
    class_path = DATA_DIR / class_name
    
    if not class_path.exists():
        print(f"Folder not found: {class_path}")
        continue
    
    for image_file in class_path.glob("*.jpg"):
        # Skip if already prefixed
        if image_file.name.startswith(class_name):
            continue
            
        new_name = f"{class_name}_{image_file.name}"
        new_path = class_path / new_name
        
        if DRY_RUN:
            print(f"Would rename: {image_file.name} → {new_name}")
        else:
            image_file.rename(new_path)
            print(f"Renamed: {image_file.name} → {new_name}")

print("\nDone!" if not DRY_RUN else "\nDry run complete. Set DRY_RUN = False to apply changes.")