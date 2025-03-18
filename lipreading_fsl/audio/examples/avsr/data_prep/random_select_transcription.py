import os
import random
import tkinter as tk
from tkinter import messagebox, scrolledtext
import simpleaudio as sa


def get_file_pairs(root_dir):
    """
    Walk through the root directory (and subdirectories) to find all .wav files
    that have a corresponding .txt file. Additionally, only include pairs if the
    transcript file contains between 1 and 6 words.

    Returns:
        A list of tuples: (wav_file_path, txt_file_path)
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Find all .wav files in this directory
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        for wav in wav_files:
            base_name = os.path.splitext(wav)[0]
            txt = base_name + '.txt'
            if txt in filenames:
                wav_full = os.path.join(dirpath, wav)
                txt_full = os.path.join(dirpath, txt)
                # Read the transcript and count the words.
                try:
                    with open(txt_full, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    # Split by whitespace to get words.
                    words = content.split()
                    # Filter out transcripts with less than 1 word or more than 6 words.
                    if len(words) < 1 or len(words) > 6:
                        continue  # Skip this pair.
                except Exception as e:
                    # Optionally log the error or skip this file pair.
                    continue
                pairs.append((wav_full, txt_full))
    return pairs


class AudioTranscriptionPlayer:
    def __init__(self, master, pairs):
        self.master = master
        self.pairs = pairs
        self.current_index = 0
        self.ratings = []  # Will store True for "correct" and False for "incorrect"

        master.title("Audio Transcription Review")
        # Increase the window size
        master.geometry("1600x1200")

        # Display progress at the top (e.g., "File 1 of 100")
        self.progress_label = tk.Label(master, text="", font=("Helvetica", 20))
        self.progress_label.pack(pady=20)

        # Create a scrollable text area for the transcript.
        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, font=("Helvetica", 16))
        self.text_area.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        self.text_area.configure(state='disabled')  # Make it read-only

        # Create a frame for the buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=20)

        # Play button to play the audio
        self.play_button = tk.Button(btn_frame, text="Play", command=self.play_audio, font=("Helvetica", 16), width=10)
        self.play_button.grid(row=0, column=0, padx=10)

        # Button to mark the transcription as correct
        self.correct_button = tk.Button(btn_frame, text="Correct", command=self.mark_correct, font=("Helvetica", 16), width=10)
        self.correct_button.grid(row=0, column=1, padx=10)

        # Button to mark the transcription as incorrect
        self.incorrect_button = tk.Button(btn_frame, text="Incorrect", command=self.mark_incorrect, font=("Helvetica", 16), width=10)
        self.incorrect_button.grid(row=0, column=2, padx=10)

        # Quit button to exit the application at any time
        self.quit_button = tk.Button(btn_frame, text="Quit", command=master.quit, font=("Helvetica", 16), width=10)
        self.quit_button.grid(row=0, column=3, padx=10)

        # Load the first transcript
        self.update_display()

    def update_display(self):
        """Update the progress label and display the current transcript."""
        self.progress_label.config(text=f"File {self.current_index + 1} of {len(self.pairs)}")
        self.text_area.configure(state='normal')
        self.text_area.delete(1.0, tk.END)
        transcript_file = self.pairs[self.current_index][1]
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading transcript: {e}"
        self.text_area.insert(tk.END, content)
        self.text_area.configure(state='disabled')

    def play_audio(self):
        """Plays the current .wav file using simpleaudio."""
        wav_file = self.pairs[self.current_index][0]
        try:
            wave_obj = sa.WaveObject.from_wave_file(wav_file)
            play_obj = wave_obj.play()
            # Not waiting for play_obj.wait_done() so that the UI remains responsive.
        except Exception as e:
            messagebox.showerror("Audio Playback Error", f"Could not play audio:\n{e}")

    def mark_correct(self):
        """Mark the current file pair as correct, then move to the next pair."""
        self.ratings.append(True)
        self.next_pair()

    def mark_incorrect(self):
        """Mark the current file pair as incorrect, then move to the next pair."""
        self.ratings.append(False)
        self.next_pair()

    def next_pair(self):
        """Move to the next pair. If finished, show a summary of results."""
        self.current_index += 1
        if self.current_index >= len(self.pairs):
            total = len(self.ratings)
            correct_count = sum(self.ratings)
            percentage = (correct_count / total) * 100 if total > 0 else 0

            # Create a custom summary window that is 5 times larger than the default messagebox.
            summary_window = tk.Toplevel(self.master)
            summary_window.title("Review Complete")
            summary_window.geometry("800x600")  # Increase the final window size

            summary_text = (
                f"Review complete!\n\n"
                f"Correct: {correct_count} / {total}\n"
                f"Percentage Correct: {percentage:.2f}%"
            )

            label = tk.Label(summary_window, text=summary_text, font=("Helvetica", 24))
            label.pack(expand=True, fill="both", padx=20, pady=20)

            close_button = tk.Button(summary_window, text="Close", font=("Helvetica", 24), command=self.master.quit)
            close_button.pack(pady=20)
        else:
            self.update_display()


def main():
    # Replace with the path to your folder containing .wav and .txt files.
    root_dir = "/home/frb6002/Documents/lipread_files_wav/pretrain"

    # Retrieve all matching file pairs that meet the word count criteria.
    pairs = get_file_pairs(root_dir)
    if len(pairs) < 100:
        print("Error: Not enough file pairs found that meet the word count criteria.")
        return

    # Randomly select 100 pairs from the available file pairs.
    selected_pairs = random.sample(pairs, 100)

    # Initialize and start the Tkinter GUI.
    root = tk.Tk()
    app = AudioTranscriptionPlayer(root, selected_pairs)
    root.mainloop()


if __name__ == "__main__":
    main()
