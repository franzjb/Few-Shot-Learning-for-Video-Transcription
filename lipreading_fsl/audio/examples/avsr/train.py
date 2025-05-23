import logging
import os
from argparse import ArgumentParser

import sentencepiece as spm
from average_checkpoints import ensemble
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


def get_trainer(args):
    seed_everything(1)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        lr_monitor,
    ]
    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=10.0
    )


def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    if args.modality == "audiovisual":
        from lightning_av import AVConformerRNNTModule

        model = AVConformerRNNTModule(args, sp_model)
    else:
        from lightning import ConformerRNNTModule

        model = ConformerRNNTModule(args, sp_model)
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality",
        choices=["audio", "video", "audiovisual"],
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to LRS3 audio-visual datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to Pretraned model.",
    )
    parser.add_argument(
        "--exp-dir",
        default="./exp",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=55,
        type=int,
        help="Number of epochs to train for. (Default: 55)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        type=str,
        help="Path to the checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to use debug level for logging",
    )
    parser.add_argument(
        "--lr",
        default=8e-4,
        type=float,
        help="Learning rate (Default: 8e-4)",
        required=False
    )
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
    trainer = get_trainer(args)
    trainer.fit(model, data_module)

    ensemble(args)


if __name__ == "__main__":
    cli_main()

'''
python train.py \
    --exp-dir /home/frb6002/Documents/experiments/offline_model/ \
    --exp-name train_2 \
    --modality audiovisual \
    --mode offline \
    --root-dir /home/frb6002/Documents/lipread_files_mp4/preprocessed \
    --sp-model-path /home/frb6002/Documents/lipreading_fsl/audio/examples/avsr/data_prep/sentencepiece/spm_model3.model \
    --num-nodes 1 \
    --gpus 1
    --lr 3e-4
    
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --exp-dir /media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/experiments/offline_model_1e-4_continued \
    --exp-name train_1 \
    --modality audiovisual \
    --mode offline \
    --root-dir /media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/soi_1_cutted_prepro \
    --sp-model-path /home/frb6002/Documents/lipreading_fsl/audio/examples/avsr/data_prep/sentencepiece/spm_model3.model \
    --resume-from-checkpoint /home/frb6002/Documents/experiments/offline_model_lr_1e-4/train_1/last.ckpt \
    --num-nodes 1 \
    --lr 1e-4 \
    --gpus 1 > /media/frb6002/e788e444-2cfa-4c6c-9b1b-16400370567d/home/audi/Documents/experiments/offline_model_1e-4_continued &


'''