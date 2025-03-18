import os
import subprocess

# Directory containing your .wav files
directory = "/home/frb6002/Documents/lipread_files_wav/comparison"

# Set the desired beginning delay in milliseconds (e.g., 200 for 200ms)
begin_delay_ms = 200

# Set the desired silence duration at the end in seconds (e.g., 0.2 for 200ms)
end_silence_sec = 0.2

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Process only .wav files that are NOT already padded (i.e. not ending with '_pad.wav')
    if filename.endswith(".wav") and not filename.endswith("_pad.wav"):
        input_path = os.path.join(directory, filename)
        basename, _ = os.path.splitext(filename)
        output_filename = f"{basename}_pad_b_e.wav"
        output_path = os.path.join(directory, output_filename)

        print(f"Processing: {input_path} -> {output_path}")

        # Construct the FFmpeg command:
        # - The 'adelay=200' filter adds 200ms of silence at the beginning (for mono files).
        # - The 'apad=pad_dur=0.2' filter appends 0.2 seconds of silence at the end.
        # - The '-y' flag forces FFmpeg to overwrite any existing output files.
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", input_path,
            "-af", f"adelay={begin_delay_ms},apad=pad_dur={end_silence_sec}",
            output_path
        ]

        subprocess.run(command)
