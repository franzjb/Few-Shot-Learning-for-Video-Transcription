import os
import re
import shutil

# Paths
INPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/sentence_of_interest"
SORTED_OUT_DIR = os.path.join(INPUT_DIR, "sorted_out")  # Destination folder

# Ensure the output directory exists
os.makedirs(SORTED_OUT_DIR, exist_ok=True)

def is_matching_format(text):
    """
    Check if the text matches the format of space-separated characters and numbers.
    """
    pattern = r"^[a-zA-Z0-9](?: [a-zA-Z0-9])*$"
    return bool(re.match(pattern, text.strip()))

def move_matching_files(directory):
    """
    Search for .txt files matching the format and move them along with corresponding .mp4 files.
    """
    moved_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                mp4_path = txt_path.replace(".txt", ".mp4")  # Corresponding video file

                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if is_matching_format(content):
                            # Move .txt file
                            shutil.move(txt_path, os.path.join(SORTED_OUT_DIR, file))
                            print(f"✅ Moved: {txt_path}")

                            # Move .mp4 file if it exists
                            if os.path.exists(mp4_path):
                                shutil.move(mp4_path, os.path.join(SORTED_OUT_DIR, os.path.basename(mp4_path)))
                                print(f"✅ Moved: {mp4_path}")

                            moved_files += 1

                except Exception as e:
                    print(f"⚠️ Error processing {txt_path}: {e}")

    if moved_files == 0:
        print("❌ No matching files found.")

# Run the script
move_matching_files(INPUT_DIR)
