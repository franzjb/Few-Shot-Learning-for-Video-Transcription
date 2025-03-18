import os
import torch
import whisper
from tqdm import tqdm

# 📂 Set the directory where your .wav files are located
AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav/comparison/"

# 📜 Get all .wav files in the folder
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith("_pad.wav")]
audio_files.sort()  # Sort for consistent processing order

# ✅ Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

# 🔄 Load Whisper Model
print("🔄 Loading Whisper model...")
model = whisper.load_model("small", device=device)

if not audio_files:
    print("❌ No audio files found! Exiting.")
    exit()

# 🏃‍♂️ Transcribe Audio and Save as .txt
for audio_file in tqdm(audio_files, desc="📜 Transcribing"):
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    txt_path = os.path.join(AUDIO_DIR, audio_file.replace(".wav", "small_200.txt"))

    # Skip if transcript already exists
    if os.path.exists(txt_path):
        print(f"⚠️ Skipping {audio_file}, transcription already exists.")
        continue

    try:
        print(f"🎤 Transcribing: {audio_file}...")

        with torch.no_grad():  # Disable gradients for inference
            result = model.transcribe(audio_path, language="de")

        # Save transcript
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"✅ Transcription saved -> {txt_path}")

    except Exception as e:
        print(f"⚠️ Error processing {audio_file}: {e}")

print("🎉 Transcription completed!")
