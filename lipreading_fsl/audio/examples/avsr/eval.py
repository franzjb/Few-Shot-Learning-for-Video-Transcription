import logging
from argparse import ArgumentParser
from email.policy import default

import sentencepiece as spm
import torch
import torchaudio
from transforms import get_data_module
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def compute_word_level_distance_filtered(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().replace(".", "").split(), seq2.lower().replace(".", "").split())


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
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][-1]
            predicted = model(batch)
            print("Actual:", actual, "Predicted:", predicted)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_edit_distance_filtered += compute_word_level_distance_filtered(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
                logger.warning(f"Filtered Processed elem {idx}; WER: {total_edit_distance_filtered / total_length}")


    logger.warning(f"Final WER: {total_edit_distance / total_length}")
    logger.warning(f"Filtered Final WER: {total_edit_distance_filtered / total_length}")
    return total_edit_distance / total_length


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality",
        required=False,
        default="audiovisual"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=False,
        default="offline"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to LRS3 audio-visual datasets.",
        required=False,
        default="/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted_prepro"
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to sentencepiece model.",
        required=False,
        default="/home/frb6002/Documents/lipreading_fsl/audio/examples/avsr/data_prep/sentencepiece/spm_model3.model"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to a checkpoint model.",
        required=False,
        default="/media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/experiments/offline_model_8e-4/train_1/last.ckpt"
    )
    parser.add_argument(
        "--lr",
        default=8e-4,
        type=float,
        help="Learning rate (Default: 8e-4)",
        required=False
    )
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

'''
python eval.py --modality=audiovisual \
               --mode=online \
               --root-dir=/home/frb6002/Documents/lipread_files_mp4/preprocessed \
               --sp-model-path=/home/frb6002/Documents/lipreading_fsl/audio/examples/avsr/data_prep/sentencepiece/spm_model3.model \
               --checkpoint-path=/home/frb6002/Documents/experiments/online_model/train_1/last.ckpt 
'''