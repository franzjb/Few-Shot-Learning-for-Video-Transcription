import os
import re
import subprocess
import string
import cv2
import multiprocessing
import glob
from textgrid import TextGrid

# Paths (Adjust as needed)
VIDEO_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/Glips_mp4"
TEXTGRID_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/textgrid"
TITLELIST_PATH = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/titlelist.txt"
OUTPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/word_of_interest"
BACKUP_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/downloaded_vids"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Set number of CPUs to use
TOTAL_CPUS = 12
RESERVED_CPUS = 2
NUM_WORKERS = max(1, TOTAL_CPUS - RESERVED_CPUS)

print(f"⚙️ Using {NUM_WORKERS} CPUs for processing, reserving {RESERVED_CPUS} for system operations.")


def clean_word(word):
    """Removes punctuation and converts word to lowercase for comparison."""
    return word.strip().lower().translate(str.maketrans("", "", string.punctuation))


def get_video_duration(video_path):
    """Returns the duration of a video file in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Unable to open video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else None


def get_word_timestamps(textgrid_path):
    """Extracts all words and timestamps from a TextGrid file."""
    try:
        tg = TextGrid.fromFile(textgrid_path)
    except Exception as e:
        print(f"⚠️ Error reading {textgrid_path}: {e}")
        return {}

    word_timestamps = {}
    for tier in tg.tiers:
        for interval in tier.intervals:
            cleaned_word = clean_word(interval.mark)
            if cleaned_word:
                if cleaned_word not in word_timestamps:
                    word_timestamps[cleaned_word] = []
                word_timestamps[cleaned_word].append((interval.minTime, interval.maxTime))

    return word_timestamps


def search_youtube(title):
    """Search for a video from Hessischer Landtag"""
    print(f"🔎 Searching YouTube for: {title}")

    try:
        search_command = [
            "yt-dlp",
            "--default-search", "ytsearch",
            "--flat-playlist",
            f"ytsearch:{title}",
            "--print", "url"
        ]
        result = subprocess.run(search_command, capture_output=True, text=True, check=True)
        urls = result.stdout.strip().split("\n")

        if urls:
            print(f"✅ Found YouTube video(s): {urls}")
            return urls[0]  # Return the first URL
        else:
            print("❌ No YouTube video found.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error searching YouTube: {e}")
        return None


def download_video(video_id, video_title):
    """Check if the video exists locally; if not, search and download from YouTube."""
    expected_filename = f"{video_id}.mp4"
    expected_path = os.path.join(BACKUP_DIR, expected_filename)

    # Check if the video already exists
    existing_files = glob.glob(os.path.join(BACKUP_DIR, f"{video_id}*.mp4"))
    if existing_files:
        print(f"✅ Video already exists: {existing_files[0]}")
        return existing_files[0]

    youtube_url = search_youtube(video_title)
    if not youtube_url:
        print(f"❌ No valid URL found for: {video_title}")
        return None

    print(f"📥 Downloading {video_title} from {youtube_url}...")

    try:
        command = [
            "yt-dlp", "-f", "best",
            "-o", expected_path,
            youtube_url
        ]
        print(f"🔧 Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(expected_path):
            print(f"✅ Successfully downloaded: {expected_path}")
            return expected_path
        else:
            print(f"❌ Download failed for {youtube_url}. Error:\n{result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading video: {e}")
        return None


def process_video(video_id, video_title):
    """Processes a single video, extracting words as separate clips."""
    video_path = download_video(video_id, video_title)
    if not video_path:
        print(f"🚫 Skipping {video_title}: Could not download video.")
        return

    textgrid_file = os.path.join(TEXTGRID_DIR, f"{video_id}.TextGrid")
    if not os.path.exists(textgrid_file):
        print(f"🚫 Skipping {video_title}: No matching TextGrid file found.")
        return

    timestamps = get_word_timestamps(textgrid_file)
    if not timestamps:
        print(f"❌ No timestamps found in {textgrid_file}.")
        return

    video_duration = get_video_duration(video_path)
    if video_duration is None:
        print(f"⚠️ Skipping {video_path} due to missing duration information.")
        return

    for word, times in timestamps.items():
        for start_time, end_time in times:
            clip_duration = end_time - start_time

            if start_time < 0 or start_time > video_duration:
                print(f"🚨 Skipping extraction: Start time ({start_time:.2f}s) out of bounds.")
                continue

            output_subdir = os.path.join(OUTPUT_DIR, video_id)
            os.makedirs(output_subdir, exist_ok=True)

            output_filename = f"{word}_{video_id}_{int(start_time * 1000)}.mp4"
            output_txt = f"{word}_{video_id}_{int(start_time * 1000)}.txt"
            output_video_path = os.path.join(output_subdir, output_filename)
            output_text_path = os.path.join(output_subdir, output_txt)

            print(f"🎬 Extracting '{word}' from {video_path}")
            print(f"   Start Time: {start_time:.2f} sec, Duration: {clip_duration:.2f} sec")
            print(f"   Output Video: {output_video_path}")
            print(f"   Output Text: {output_text_path}")

            ffmpeg_cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time), "-t", str(clip_duration),
                "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "128k",
                output_video_path
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with open(output_text_path, "w") as txt_file:
                txt_file.write(word)


if __name__ == "__main__":
    with open(TITLELIST_PATH, "r") as f:
        tasks = [re.match(r"(\d+)=(.+)", line.strip()).groups() for line in f if re.match(r"(\d+)=(.+)", line.strip())]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pool.starmap(process_video, tasks)
