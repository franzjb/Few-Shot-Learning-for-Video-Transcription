import os
import re

#Root directory to search for .txt files
root_dir = "/home/frb6002/Documents/lipread_files_wav"

#Output file path
output_file = "/home/frb6002/Documents/lipread_files_wav/merged_txt_files.txt"

#Expression to remove special characters and dots
clean_text = lambda text: re.sub(r"[^a-zA-Z0-9äöüÄÖÜß\s]", "", text)

with open(output_file, "w", encoding="utf-8") as outfile:

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        cleaned_line = clean_text(line.strip())
                        if cleaned_line:
                            outfile.write(cleaned_line + "\n")

print(f"All .txt files from {root_dir} and subdirectories merged into {output_file}.")
