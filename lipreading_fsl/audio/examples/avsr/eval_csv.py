import logging
import csv
import re
from argparse import ArgumentParser
import sentencepiece as spm
import torch
import torchaudio
from transforms import get_data_module
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def clean_transcription(seq):
    # Define a regex pattern to remove unwanted special characters
    pattern = r"[.,;!?`\"-]"
    return re.sub(pattern, "", seq.lower())


def compute_word_level_distance_filtered(seq1, seq2):
    seq1_cleaned = clean_transcription(seq1)
    seq2_cleaned = clean_transcription(seq2)
    return torchaudio.functional.edit_distance(seq1_cleaned.split(), seq2_cleaned.split())


def contains_long_word(seq):
    return any(len(word) > 25 for word in seq.split())


def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    if args.modality == "audiovisual":
        from lightning_av import AVConformerRNNTModule
        model = AVConformerRNNTModule(args, sp_model)
    else:
        from lightning import ConformerRNNTModule
        model = ConformerRNNTModule(args, sp_model)

    ckpt = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def run_eval(model, data_module):
    total_edit_distance = 0
    total_edit_distance_filtered = 0
    total_length = 0
    dataloader = data_module.test_dataloader()
    csv_file_path = "/home/frb6002/Documents/experiments/offline_model_lr_1e-4/train_1/comparison.csv"

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Actual Transcription", "Predicted Transcription", "Filtered Actual", "Filtered Predicted",
             "WER", "Filtered WER"])

        with torch.no_grad():
            total_length_filtered = 0  # Track length of filtered transcripts

            for idx, (batch, sample) in enumerate(dataloader):
                actual = sample[0][-1]
                predicted = model(batch)

                # Clean transcriptions
                filtered_actual = clean_transcription(actual)
                filtered_predicted = clean_transcription(predicted)

                # Compute WER
                wer = compute_word_level_distance(actual, predicted) / len(actual.split())
                filtered_wer = compute_word_level_distance_filtered(filtered_actual, filtered_predicted) / len(
                    filtered_actual.split()) if len(filtered_actual.split()) > 0 else 0

                writer.writerow([actual, predicted, filtered_actual, filtered_predicted, wer, filtered_wer])

                # Accumulate total edit distance and lengths
                total_edit_distance += compute_word_level_distance(actual, predicted)
                total_edit_distance_filtered += compute_word_level_distance_filtered(filtered_actual,
                                                                                     filtered_predicted)
                total_length += len(actual.split())
                total_length_filtered += len(filtered_actual.split())

                if idx % 100 == 0:
                    logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
                    logger.warning(
                        f"Filtered Processed elem {idx}; WER: {total_edit_distance_filtered / total_length_filtered if total_length_filtered > 0 else 0}")

    logger.warning(f"Final WER: {total_edit_distance / total_length}")
    logger.warning(f"Filtered Final WER: {total_edit_distance_filtered / total_length}")
    return total_edit_distance / total_length


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--modality", type=str, default="audiovisual", help="Modality")
    parser.add_argument("--mode", type=str, default="offline", help="Perform online or offline recognition.")
    parser.add_argument("--root-dir", type=str, default="/home/frb6002/Documents/lipread_files_mp4/preprocessed",
                        help="Root directory to LRS3 audio-visual datasets.")
    parser.add_argument("--sp-model-path", type=str,
                        default="/home/frb6002/Documents/lipreading_fsl/audio/examples/avsr/data_prep/sentencepiece/spm_model3.model",
                        help="Path to sentencepiece model.")
    parser.add_argument("--checkpoint-path", type=str,
                        default="/home/frb6002/Documents/experiments/offline_model_lr_1e-4/train_1/last.ckpt",
                        help="Path to a checkpoint model.")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate (Default: 8e-4)")
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    data_module = get_data_module(args, str(args.sp_model_path))
    run_eval(model, data_module)


if __name__ == "__main__":
    cli_main()
