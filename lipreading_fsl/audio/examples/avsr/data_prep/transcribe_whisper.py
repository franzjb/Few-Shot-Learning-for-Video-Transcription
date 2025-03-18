import os
import torch
import whisper
from tqdm import tqdm

# 📝 Get .wav Files (Recursively Search in Subfolders)
#AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav/comparison/"
#AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav/comparison_test1/aber"
#AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav/comparison_test1/bürgerinnen"
AUDIO_DIR = "/home/frb6002/Documents/lipread_files_wav/trainval"

audio_files = [os.path.join(root, file) for root, _, files in os.walk(AUDIO_DIR) for file in files if file.endswith(".wav")]
audio_files.sort()

# ✅ Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

# 🔄 Load Whisper Model
print("🔄 Loading Whisper Small model...")
model = whisper.load_model("small", device=device)

if not audio_files:
    print("❌ No audio files found! Exiting.")
    exit()


WHISPER_CONFIGS = {
    "temperature": 0.2,  # 0.0 fully determistic - 1.0 high randomness - Reduce randomness, lower more stable, higher more variation
    "word_timestamps": False,  # Improve word alignment
    "compression_ratio_threshold": 2.4,  # 1.0 very strict text removal - 5.0 keeps everything - Reduce hallucinations, lower values remove more vad text
    "logprob_threshold": -1.0,  # -inf keeps everything - -0.0 strict filtering - Remove uncertain text, higher values cut out more text
    "condition_on_previous_text": False, # Avoid contextual bias, True uses context
    "language": "de", # Force to German
    "beam_size": 7, # 1 fastest - 10 most accurate - number of hypotheses, higher more accurate but slower
    "patience": 2, # 1.0 - 2.0 very careful selection - spend more time selecting best hypothesis, higher more accurate but slower
    "best_of": 7 # 1 fastest - 20 most exhaustive search - multiple outputs selects the best, 10 for highest accuracy
}
# 🏃‍♂️ Transcribe Audio
for audio_file in tqdm(audio_files, desc="📜 Transcribing"):
    print(f"🎤 Transcribing: {os.path.basename(audio_file)}...")
    result = model.transcribe(audio_file, **WHISPER_CONFIGS)

    # Save transcript
    #transcript_path = audio_file.replace(".wav", "_improved_small_b_e.txt")
    transcript_path = audio_file.replace(".wav", ".txt")
    with open(transcript_path, "w") as f:
        f.write(result["text"])

    #print(f"✅ Transcription saved -> {transcript_path}")
    print(f"✅ Transcription saved -> {transcript_path}", "Length", len(result["text"].split(' ')))
print("🎉 Transcription completed!")

'''
CUDA_VISIBLE_DEVICES=0 python transcribe_whisper.py /home/frb6002/Documents/lipread_files_wav/trainval 0 6250 > gpu_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python transcribe_whisper.py /home/frb6002/Documents/lipread_files_wav/trainval 6250 12500 > gpu_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python transcribe_whisper.py /home/frb6002/Documents/lipread_files_wav/trainval 12500 18750 > gpu_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python transcribe_whisper.py /home/frb6002/Documents/lipread_files_wav/trainval 18750 25000 > gpu_3.log 2>&1 &
'''