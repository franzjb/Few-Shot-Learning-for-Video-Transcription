import os
import torch
import whisper
from tqdm import tqdm

# 📝 Get .wav Files (Recursively Search in Subfolders)
AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav"
audio_files = [os.path.join(root, file) for root, _, files in os.walk(AUDIO_DIR) for file in files if
               file.endswith(".wav")]
audio_files.sort()

# ✅ Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

# 🔄 Load Whisper Model
print("🔄 Loading Whisper Turbo model...")
model = whisper.load_model("large", device=device)

if not audio_files:
    print("❌ No audio files found! Exiting.")
    exit()


def remove_already_processed(files_list):
    updated_files_list = []
    for each_file in files_list:
        if not os.path.isfile(each_file.replace(".wav", ".txt")):
            updated_files_list.append(each_file)
    return updated_files_list

files_to_process = remove_already_processed(audio_files)


print(len(files_to_process))
# # 🏃‍♂️ Transcribe Audio
# for audio_file in tqdm(audio_files, desc="📜 Transcribing"):
#     print(f"🎤 Transcribing: {os.path.basename(audio_file)}...")
#     result = model.transcribe(audio_file)
#
#     # Save transcript
#     transcript_path = audio_file.replace(".wav", ".txt")
#     with open(transcript_path, "w") as f:
#         f.write(result["text"])
#
#     print(f"✅ Transcription saved -> {transcript_path}")
# print("🎉 Transcription completed!")

