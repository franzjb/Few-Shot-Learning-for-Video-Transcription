import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Directories
INPUT_DIR = Path("/home/frb6002/Documents/lipread_files/")
OUTPUT_DIR = Path("/home/frb6002/Documents/lipread_files_wav/")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all .m4a files recursively
m4a_files = list(INPUT_DIR.rglob("*.m4a"))

# Filter out files that are already converted
m4a_files = [f for f in m4a_files if not (OUTPUT_DIR / f.relative_to(INPUT_DIR)).with_suffix(".wav").exists()]

if not m4a_files:
    print("All files are already converted. Nothing to do!")
    exit()

def convert_to_lrs3_wav(m4a_path):
    """Convert .m4a to .wav in LRS3 format (16-bit PCM, 16kHz, mono)"""
    rel_path = m4a_path.relative_to(INPUT_DIR)  # Keep relative path
    wav_path = OUTPUT_DIR / rel_path.with_suffix(".wav")  # Target file path

    # Create parent directories if needed
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command for LRS3 conversion
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite existing files
        "-i", str(m4a_path),  # Input file
        "-ac", "1",  # Mono audio
        "-ar", "16000",  # Resample to 16kHz
        "-sample_fmt", "s16",  # 16-bit PCM (LRS3 format)
        str(wav_path)
    ]

    # Run the command quietly (suppress stdout/stderr)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return m4a_path  # Return for progress bar update

# Use multiprocessing with a progress bar
with tqdm(total=len(m4a_files), desc="Converting", unit="file") as pbar:
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(convert_to_lrs3_wav, file): file for file in m4a_files}
        
        for future in as_completed(futures):
            _ = future.result()  # Retrieve result to catch errors
            pbar.update(1)  # Increment progress bar

print(f"Conversion completed! WAV files saved in: {OUTPUT_DIR}")
