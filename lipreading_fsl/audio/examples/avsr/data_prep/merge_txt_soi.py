import os
import shutil

# Set source and destination folders
source_folder = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1"  # Change this
destination_folder = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted"  # Change this

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Walk through the source folder
for root, _, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".txt"):  # Only process .txt files
            # Get the relative path from source folder
            relative_path = os.path.relpath(root, source_folder)
            target_dir = os.path.join(destination_folder, relative_path)

            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)

            # Copy the file while keeping the structure
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, file)
            shutil.copy2(source_file, destination_file)

            print(f"Copied: {source_file} → {destination_file}")
