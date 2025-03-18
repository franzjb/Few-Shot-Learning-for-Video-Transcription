import os
import subprocess
import glob
import multiprocessing
from tqdm import tqdm  # Progress bar for better tracking

# Paths
INPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1"
OUTPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Crop and Zoom Values
CROP_WIDTH = 256  # Final width after cropping
CROP_HEIGHT = 256  # Final height after cropping
CROP_X = 250  # Shift crop region left/right
CROP_Y = 75  # Shift crop region up/down
ZOOM_FACTOR = 1.5  # 1.0 = No zoom, >1.0 zooms in, <1.0 zooms out

# Auto-detect the number of available CPUs
NUM_CORES = os.cpu_count()  # Detects all cores
NUM_WORKERS = max(1, NUM_CORES - 1)  # Leave 1 core free for system tasks

print(f"⚙️ Using {NUM_WORKERS} CPU cores for processing videos.")

def crop_video(video_path):
    """Crops and zooms the video while preserving the folder structure."""
    filename = os.path.relpath(video_path, start=INPUT_DIR)  # Keep subfolder structure
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Overwrite existing file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"♻️ Overwriting existing file: {output_path}")

    # Adjust crop values based on zoom
    zoomed_x = int(CROP_X * ZOOM_FACTOR)
    zoomed_y = int(CROP_Y * ZOOM_FACTOR)

    # FFmpeg command with zoom & crop
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale=iw*{ZOOM_FACTOR}:ih*{ZOOM_FACTOR}, crop={CROP_WIDTH}:{CROP_HEIGHT}:{zoomed_x}:{zoomed_y}",
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    print(f"▶️ Processing: {filename}")

    # Run FFmpeg and capture output
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"❌ FFmpeg Error processing {filename}: {result.stderr}")
    else:
        print(f"✅ Cropped & Zoomed video saved: {output_path}")


if __name__ == "__main__":
    # Search all .mp4 files recursively
    video_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True)

    if not video_files:
        print(f"❌ No .mp4 files found in {INPUT_DIR}. Check the path and try again.")
    else:
        print(f"✅ Found {len(video_files)} video files. Processing...")

    # Process videos with progress tracking
    for video in tqdm(video_files, desc="Cropping & Zooming Videos"):
        crop_video(video)
