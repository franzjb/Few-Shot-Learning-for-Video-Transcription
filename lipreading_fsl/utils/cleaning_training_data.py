import os
import shutil


def shift_unusable_files(input_folder, target_folder):
    # Create the target folder if it does not exist
    os.makedirs(target_folder, exist_ok=True)

    # Traverse through the input folder and its subfolders
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Process only .txt files
            if file.lower().endswith(".txt"):
                txt_path = os.path.join(root, file)
                try:
                    # Open and read the content of the .txt file
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
                    continue

                # Count the words by splitting on whitespace
                word_count = len(content.split())

                # Check the condition: if the .txt file is empty or has more than 6 words
                if word_count == 0 or word_count > 6:
                    # Construct new destination paths for the .txt file
                    new_txt_path = os.path.join(target_folder, file)
                    print(f"Moving {txt_path} (word count: {word_count}) to {new_txt_path}")

                    try:
                        shutil.move(txt_path, new_txt_path)
                    except Exception as e:
                        print(f"Error moving {txt_path}: {e}")

                    # Now find the corresponding .mp4 file
                    base_name = os.path.splitext(file)[0]
                    mp4_filename = base_name + ".mp4"
                    mp4_path = os.path.join(root, mp4_filename)

                    # If the corresponding .mp4 exists, move it as well
                    if os.path.exists(mp4_path):
                        new_mp4_path = os.path.join(target_folder, mp4_filename)
                        print(f"Moving corresponding video {mp4_path} to {new_mp4_path}")
                        try:
                            shutil.move(mp4_path, new_mp4_path)
                        except Exception as e:
                            print(f"Error moving {mp4_path}: {e}")
                    else:
                        print(f"Corresponding video not found: {mp4_path}")


if __name__ == "__main__":
    # Define the input folder and target folder paths
    input_folder = "/home/frb6002/Documents/lipread_files_mp4/trainval"
    target_folder = "/home/frb6002/Documents/lipread_files_unusable"

    shift_unusable_files(input_folder, target_folder)
