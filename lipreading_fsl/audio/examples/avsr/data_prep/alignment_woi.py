import os
import glob
import subprocess
import torch
import torchaudio
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import whisper

# ----------- CONFIGURATIONS ------------
INPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/Glips_mp4"
OUTPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/preprocessed_aligned"
MISSING_WORDS_FILE = os.path.join(OUTPUT_DIR, "missing_words.log")
USE_WHISPER = False  # ✅ Toggle between Whisper (True) and Wav2Vec2 (False)

# ----------- DEVICE CONFIGURATION ------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- LOAD ASR MODEL ------------
if USE_WHISPER:
    asr_model = whisper.load_model("large-v2").to(device)  # ✅ Whisper Large-v2
else:
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)

# Ensure model is in eval mode
asr_model.eval()


# ----------- STEP 1: Find All Videos ------------
def find_videos(input_dir):
    return glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)


video_files = find_videos(INPUT_DIR)
print(f"Found {len(video_files)} video files.")


# ----------- STEP 2: Extract Word from Filename ------------
def extract_word_from_filename(video_path):
    filename = os.path.basename(video_path)
    word = filename.split("_")[0].lower()
    return ''.join(e for e in word if e.isalnum())


# ----------- STEP 3: Get Output Path ------------
def get_output_path(video_path, word):
    relative_path = os.path.relpath(video_path, INPUT_DIR)
    folder_structure = os.path.dirname(relative_path)

    output_folder = os.path.join(OUTPUT_DIR, folder_structure, word)
    os.makedirs(output_folder, exist_ok=True)

    output_video_path = os.path.join(output_folder, os.path.basename(video_path))
    txt_file_path = output_video_path.replace(".mp4", ".txt")

    return output_video_path, txt_file_path


# ----------- STEP 4: Convert Video to Audio and Apply Denoising ------------
def extract_denoised_audio(video_path, target_sr=16000):
    """Extracts audio from the video, applies noise reduction, and returns a cleaned numpy array."""
    audio_path = video_path.replace(".mp4", ".wav")

    # Extract audio using ffmpeg
    subprocess.run(["ffmpeg", "-i", video_path, "-ar", str(target_sr), "-ac", "1", "-y", audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load and denoise audio
    waveform, _ = librosa.load(audio_path, sr=target_sr)
    reduced_noise = nr.reduce_noise(y=waveform, sr=target_sr)

    return reduced_noise, target_sr


# ----------- STEP 5: ASR Transcription ------------
def transcribe_audio(audio_array, sample_rate):
    """Transcribes audio using Whisper or Wav2Vec2."""
    if USE_WHISPER:
        temp_audio_path = "/tmp/temp_audio.wav"
        sf.write(temp_audio_path, audio_array, sample_rate)
        result = asr_model.transcribe(temp_audio_path, language="de")
        return result["text"].upper()
    else:
        inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = asr_model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].upper()


# ----------- STEP 6: Forced Alignment ------------
def align_text_with_audio(audio_array, sample_rate, word_of_interest):
    transcript = transcribe_audio(audio_array, sample_rate)
    print(f"🔎 ASR Model Predicted: {transcript}")

    if word_of_interest.upper() not in transcript:
        print(f"⚠️ Word '{word_of_interest}' not found in ASR output.")
        return None, None

    start_time = 0.2  # Dummy start time (since we need real forced alignment)
    end_time = start_time + 1.0  # Dummy end time

    if end_time - start_time <= 0:
        print(f"⚠️ Invalid timestamps for '{word_of_interest}' (start: {start_time}, end: {end_time})")
        return None, None

    return start_time, end_time


# ----------- STEP 7: Trim the Video Based on Alignment ------------
def trim_video(video_path, start_time, end_time, output_path):
    """Cuts video using FFmpeg based on timestamps, handling errors."""
    if start_time is None or end_time is None or end_time <= start_time:
        print(f"⚠️ Skipping {video_path} due to invalid timestamps.")
        return

    command = [
        "ffmpeg", "-i", video_path, "-ss", str(start_time), "-to", str(end_time),
        "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", output_path, "-y"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"⚠️ FFmpeg failed for {video_path}: {result.stderr.decode()}")


# ----------- STEP 8: Process All Videos ------------
for video_path in tqdm(video_files, desc="Processing Videos"):
    word_of_interest = extract_word_from_filename(video_path)
    output_video_path, txt_file_path = get_output_path(video_path, word_of_interest)

    if os.path.exists(output_video_path) and os.path.exists(txt_file_path):
        print(f"Skipping {video_path}, already processed.")
        continue

    # Extract and denoise audio
    audio_array, sample_rate = extract_denoised_audio(video_path)

    # Get exact timestamps using forced alignment
    start, end = align_text_with_audio(audio_array, sample_rate, word_of_interest)

    if start is not None and end is not None:
        trim_video(video_path, start, end, output_video_path)
        with open(txt_file_path, "w") as f:
            f.write(word_of_interest + "\n")
        print(f"✅ Trimmed video saved as {output_video_path}")
    else:
        with open(MISSING_WORDS_FILE, "a") as log_file:
            log_file.write(f"{video_path}: '{word_of_interest}' not found\n")
        print(f"⚠️ Word '{word_of_interest}' not found in {video_path}")
