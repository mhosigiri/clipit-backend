import os
import shutil

# Settings
SIZE_THRESHOLD_MB = 100
MOVE_INSTEAD_OF_DELETE = True
DEST_FOLDER = "large_files_backup"

# Convert threshold to bytes
SIZE_THRESHOLD = SIZE_THRESHOLD_MB * 1024 * 1024

# Step 1: Create destination folder if moving
if MOVE_INSTEAD_OF_DELETE and not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

# Step 2: Walk through files and find large ones
for root, _, files in os.walk("."):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            file_size = os.path.getsize(file_path)
            if file_size > SIZE_THRESHOLD:
                size_mb = file_size / (1024 * 1024)
                print(f"⚠️ Large file found: {file_path} ({size_mb:.2f} MB)")
                if MOVE_INSTEAD_OF_DELETE:
                    dest_path = os.path.join(DEST_FOLDER, os.path.relpath(file_path, start="."))
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(file_path, dest_path)
                    print(f"   ✅ Moved to: {dest_path}")
                else:
                    os.remove(file_path)
                    print("   ❌ Deleted.")
        except FileNotFoundError:
            continue