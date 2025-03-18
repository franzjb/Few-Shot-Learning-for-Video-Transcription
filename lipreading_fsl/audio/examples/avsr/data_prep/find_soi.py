import os
import re
import subprocess
import string
import cv2
import multiprocessing
import glob
from textgrid import TextGrid

# Paths (Updated Output Directory)
#VIDEO_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/Glips_mp4"
TEXTGRID_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/textgrid"
TITLELIST_PATH = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/titlelist.txt"
OUTPUT_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/sentence_of_interest_8iv"
BACKUP_DIR = "/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/downloaded_vids"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Constants
NUM_INTERVALS = 8
NUM_WORKERS = 1

def clean_word(word):
    """Removes punctuation and converts word to lowercase."""
    return word.strip().lower().translate(str.maketrans("", "", string.punctuation))


def get_video_duration(video_path):
    """Returns the duration of a video file in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else None


def get_word_timestamps(textgrid_path):
    """Extracts words and timestamps from the first item in a TextGrid file."""
    try:
        tg = TextGrid.fromFile(textgrid_path)

        if not tg.tiers:
            print(f"⚠️ Warning: No tiers found in {textgrid_path}")
            return []

        # ✅ Only process the first tier (ignore others like item[2])
        first_tier = tg.tiers[0]

        intervals = []
        for interval in first_tier.intervals:
            cleaned_word = clean_word(interval.mark)
            if cleaned_word:
                intervals.append((cleaned_word, interval.minTime, interval.maxTime))

        return intervals

    except Exception as e:
        print(f"⚠️ Error reading {textgrid_path}: {e}")
        return []


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
    """Processes a video, extracting segments every NUM_INTERVALS words."""
    video_path = download_video(video_id, video_title)
    if not video_path:
        print(f"🚫 Skipping {video_title}: Could not download video.")
        return

    textgrid_path = os.path.join(TEXTGRID_DIR, f"{video_id}.TextGrid")
    if not os.path.exists(textgrid_path):
        print(f"🚫 Skipping {video_title}: No matching TextGrid file found.")
        return

    timestamps = get_word_timestamps(textgrid_path)
    if not timestamps:
        print(f"❌ No timestamps found in {textgrid_path}.")
        return

    video_duration = get_video_duration(video_path)
    if video_duration is None:
        print(f"⚠️ Skipping {video_path} due to missing duration information.")
        return

    output_subdir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(output_subdir, exist_ok=True)

    # ✅ Skip if the video folder exists and has segments
    if any(glob.glob(os.path.join(output_subdir, "*.mp4"))):
        print(f"⏩ Skipping {video_id}, already processed.")
        return

    # Process in groups of NUM_INTERVALS
    for i in range(0, len(timestamps), NUM_INTERVALS):
        start_time = timestamps[i][1]
        end_time = timestamps[min(i + NUM_INTERVALS, len(timestamps) - 1)][2]
        text_content = " ".join(word for word, _, _ in timestamps[i:i + NUM_INTERVALS])

        output_video_path = os.path.join(output_subdir, f"{video_id}_{i}_{i + NUM_INTERVALS}.mp4")
        output_text_path = os.path.join(output_subdir, f"{video_id}_{i}_{i + NUM_INTERVALS}.txt")

        # ✅ Skip if segment files already exist
        if os.path.exists(output_video_path) and os.path.exists(output_text_path):
            print(f"⏩ Skipping segment {output_video_path}, already exists.")
            continue

        print(f"🎬 Extracting segment from {start_time:.2f} sec to {end_time:.2f} sec")
        print(f"   Text: {text_content}")
        print(f"   Output Video: {output_video_path}")
        print(f"   Output Text: {output_text_path}")

        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time), "-to", str(end_time),
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "128k",
            output_video_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(output_text_path, "w") as txt_file:
            txt_file.write(text_content)


# Define the starting title number (Modify this)
START_FROM = 300  # This means it will start from title 0301 and onward

if __name__ == "__main__":
    with open(TITLELIST_PATH, "r") as f:
        all_tasks = [re.match(r"(\d+)=(.+)", line.strip()).groups() for line in f if re.match(r"(\d+)=(.+)", line.strip())]

    # ✅ Convert title numbers to integers and filter those greater than START_FROM
    tasks = [(video_id, title) for video_id, title in all_tasks if int(video_id) > START_FROM]

    if not tasks:
        print(f"❌ No titles found after {START_FROM}.")
    else:
        print(f"✅ Processing {len(tasks)} videos starting from title {START_FROM + 1}.")

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pool.starmap(process_video, tasks)
