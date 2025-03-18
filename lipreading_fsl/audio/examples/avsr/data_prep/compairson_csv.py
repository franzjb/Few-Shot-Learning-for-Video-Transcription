import os
import pandas as pd
import re
from collections import defaultdict

# 📂 Set the directory where transcribed .txt files are stored
TXT_DIR = "/home/frb6002/Documents/lipread_files_wav/comparison/"

# 🏷️ Valid Whisper model names
WHISPER_MODELS = ["small", "tiny", "turbo", "large", "base", "medium"]

# 🏷️ Extract base filename and model type from filenames
def extract_info(filename):
    match = re.match(r"(.*?)-(\d+)(base|large|medium|small|tiny|turbo)\.txt$", filename)
    if match:
        base_name, number, model_name = match.groups()
        return f"{base_name}-{number}", model_name
    return None, None

# 📜 Get all transcribed .txt files that match the correct naming format
txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]

# 🏷️ Organize files into {base_filename: {model_name: filepath}}
grouped_files = defaultdict(dict)

for file in txt_files:
    base_name, model_name = extract_info(file)
    if base_name and model_name:
        grouped_files[base_name][model_name] = os.path.join(TXT_DIR, file)

# 🏁 Ensure we have valid transcriptions to compare
if not grouped_files:
    print("❌ No correctly named transcriptions found! Check your filenames.")
    exit()

# 📊 Create a DataFrame for each base filename and compare models
comparison_data = []

for base_name, model_files in sorted(grouped_files.items()):
    model_contents = {}

    # Read all files for this base name
    for model, filepath in model_files.items():
        with open(filepath, "r", encoding="utf-8") as f:
            model_contents[model] = f.readlines()

    # Determine the max number of lines
    max_lines = max(len(content) for content in model_contents.values())

    # Align text from different models
    for i in range(max_lines):
        row = {"Filename": base_name}
        for model in WHISPER_MODELS:  # Ensure column order
            row[model] = model_contents.get(model, [""] * max_lines)[i].strip() if i < len(model_contents.get(model, [])) else ""
        comparison_data.append(row)

# 📝 Convert to DataFrame & Save as Excel for better readability
df = pd.DataFrame(comparison_data)

output_path = os.path.join(TXT_DIR, "comparison_output.xlsx")
df.to_excel(output_path, index=False)

print(f"✅ Comparison file created: {output_path}")