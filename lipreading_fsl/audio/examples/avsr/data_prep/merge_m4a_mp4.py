import os
import subprocess#
# from tqdm import tqdm  # pip install tqdm if needed

from multiprocessing import Pool
INPUT_DIR = "/home/frb6002/Documents/lipread_files/trainval"
OUTPUT_DIR = "/home/frb6002/Documents/lipread_files_mp4/trainval"

# First, collect all tasks (video files with matching audio)
tasks = []
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            audio_path = video_path[:-4] + ".m4a"  # Replace .mp4 with .m4a
            if os.path.isfile(audio_path):
                # Compute relative output path and add to tasks
                rel_path = os.path.relpath(video_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, rel_path)
                tasks.append((video_path, audio_path, output_path))
            else:
                # Optionally, print a warning if no matching audio exists
                print(f"No matching audio for: {video_path}")

def run_parallel(index):
    video_path, audio_path, output_path = tasks[index]
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Print the current file being processed (tqdm.write ensures proper progress bar display)

    # Run FFmpeg with minimal logging:
    # - '-loglevel error' tells FFmpeg to only print errors.
    # - stdout and stderr are redirected to DEVNULL to hide them.
    subprocess.run([
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


process = Pool()
process.map(run_parallel, range(len(tasks)))

'''
parallel --jobs 8 python merge_m4a_mp4.py [{}, {}] ::: $(seq 0 3125 25000)
'''

print("Here")
# Process each file with a progress bar
# for video_path, audio_path, output_path in tqdm(tasks, desc="Merging files", unit="file"):
#     # Ensure the output folder exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     # Print the current file being processed (tqdm.write ensures proper progress bar display)
#     tqdm.write(f"Processing: {video_path}")
#
#     # Run FFmpeg with minimal logging:
#     # - '-loglevel error' tells FFmpeg to only print errors.
#     # - stdout and stderr are redirected to DEVNULL to hide them.
#     subprocess.run([
#         "ffmpeg",
#         "-loglevel", "error",
#         "-i", video_path,
#         "-i", audio_path,
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-b:a", "128k",
#         output_path
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


