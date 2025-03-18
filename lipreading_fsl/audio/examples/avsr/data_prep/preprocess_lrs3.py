import argparse
import glob
import math
import os
import shutil
import warnings
import cv2

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="LRS3 Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    help="The directory for sequence.",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Face detector used in the experiment.",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Specify the dataset name used in the experiment",
)
parser.add_argument(
    "--root-dir",
    type=str,
    help="The root directory of cropped-face dataset.",
)
parser.add_argument(
    "--subset",
    type=str,
    required=True,
    help="Subset of the dataset used in the experiment.",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Length of the segment in seconds.",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
dataset = args.dataset

args.data_dir = os.path.normpath(args.data_dir)
vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, resize=(96, 96))
aud_dataloader = AVSRDataLoader(modality="audio")

# Create label file
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{dataset}_{args.subset}_transcript_lengths_seg{args.seg_duration}s.csv"
    if args.groups <= 1
    else f"{dataset}_{args.subset}_transcript_lengths_seg{args.seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"📂 Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")

# Output directories
dst_vid_dir = os.path.join(args.root_dir, dataset, dataset + f"_video_seg{args.seg_duration}s")
dst_txt_dir = os.path.join(args.root_dir, dataset, dataset + f"_text_seg{args.seg_duration}s")

# Search for video files
if args.subset == "test":
    filenames = glob.glob(os.path.join(args.data_dir, args.subset, "**", "*.mp4"), recursive=True)
elif args.subset == "train":
    filenames = glob.glob(os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True)
    filenames.extend(glob.glob(os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True))
    filenames.sort()
else:
    raise NotImplementedError("Subset must be 'train' or 'test'.")

print(f"✅ Found {len(filenames)} videos to process in {args.data_dir}")

# Parallel job distribution
unit = math.ceil(len(filenames) / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]
print(f"📂 {len(filenames)} videos assigned to this job.")

# Process each video
for data_filename in tqdm(filenames, desc="Processing Videos"):
    try:
        print(f"🎥 Processing file: {data_filename}")
        video_data = vid_dataloader.load_data(data_filename)
        audio_data = aud_dataloader.load_data(data_filename)

        if video_data is None or len(video_data) == 0:
            print(f"⚠️ Skipping {data_filename}: No video data!")
            continue

        if audio_data is None or audio_data.size(1) == 0:
            print(f"⚠️ Skipping {data_filename}: No audio data!")
            continue

        # ✅ Fix: Define segment lengths AFTER loading video
        video_length = len(video_data)
        audio_length = audio_data.size(1)

        print(f"🎬 Video frames: {video_length}, 🎵 Audio samples: {audio_length}")

        expected_ratio = 640  # Approximate expected ratio
        tolerance = 200  # Allow flexibility

        actual_ratio = audio_length / video_length
        if abs(actual_ratio - expected_ratio) > tolerance:
            print(f"⚠️ Unusual audio/video ratio {actual_ratio:.2f}, but proceeding")

    except Exception as e:
        print(f"❌ Can't process {data_filename}: {e}")
        continue

    dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
    dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.wav"
    dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"

    # 🛠 Fix: Load transcript from the existing .txt file
    transcript_path = data_filename[:-4] + ".txt"

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as txt_file:
            transcript_data = txt_file.read().strip()  # Read and clean transcript
    else:
        print(f"⚠️ Missing transcript file: {transcript_path}")
        transcript_data = ""  # Leave empty if not found

    # 🛠 Add Debugging Print Here
    print(f"📜 Transcript for {data_filename}: {transcript_data}")  # ✅ Debugging print statement

    # Pass the actual transcript to save function
    save_vid_aud_txt(
        dst_vid_filename,
        dst_aud_filename,
        dst_txt_filename,
        video_data,
        audio_data,
        transcript_data,  # ✅ Fix - Use the actual transcript data
        video_fps=25,
        audio_sample_rate=16000,
    )

    # Merge Video + Audio
    try:
        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"], in2["a"], dst_vid_filename[:-4] + ".m.mp4",
            vcodec="copy", acodec="aac", strict="experimental", loglevel="panic"
        )
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)
        print(f"✅ Merged video saved: {dst_vid_filename}")

    except Exception as e:
        print(f"❌ Error merging {data_filename}: {e}")

    # Save label info
    basename = os.path.relpath(dst_vid_filename, start=os.path.join(args.root_dir, dataset))
    transcript_text = transcript_data.strip() if transcript_data else "MISSING"
    f.write(f"{dataset},{basename},{video_length},{len(transcript_text)}\n")

f.close()
print("✅ Processing complete!")


'''
parallel --jobs 8 

python preprocess_lrs3.py \
    --data-dir=/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted \
    --detector=mediapipe \
    --dataset=lrs3 \
    --root-dir=/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/cut_vid_preprocessed \
    --subset=train \
    --groups 2 \
     --job-index {}
      
      ::: $(seq 0 7)

parallel --jobs 10 --progress --eta --bar \
    python preprocess_lrs3.py \
        --data-dir=/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted \
        --detector=mediapipe \
        --dataset=lrs3 \
        --root-dir=/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted_prepro \
        --subset=train \
        --groups 4 \
        --job-index {} ::: $(seq 0 9)



'''