import argparse
import datetime
import os
import random
import time


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from mnist import MNISTDataModule

from net.linear import Linear
from net.conv import Conv


# Deterministic

MODEL_DIRECTORY = {
    "Linear": Linear,
    "Conv": Conv
}


DATALOADER_DIRECTORY = {
    'MNIST': MNISTDataModule,
} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    parser.add_argument('--eval', help='Whether to test model on the best iteration after training'
                        , default=False, type=bool)
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument("--load",
                        help="Directory of pre-trained model weights only,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained checkpoint including hyperparams,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    parser.add_argument('--data_dir', help='Directory of your Dataset', required=True, default=None)
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=1, type=int)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=0.1, type=float)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units',
                        default=64, type=int)
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--pin_memory', help='Whether to utilize pin_memory in dataloader', type=bool, default=True)
    parser.add_argument('--val_ratio', help='Float between [0, 1] to indicate the percentage of train dataset to validate on', type=float, default=0.2)


    args = parser.parse_args()
    dict_args = vars(args)
    
    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    print(f"[info] Model: {dict_args['model']}")
    assert dict_args['model'] in MODEL_DIRECTORY
    if dict_args['load'] is not None:
        model = MODEL_DIRECTORY[dict_args['model']].load_from_checkpoint(dict_args['load'], **dict_args)
    else:
        model = MODEL_DIRECTORY[dict_args['model']](**dict_args)

    # Initialize logging paths
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))

    os.makedirs(weight_save_dir, exist_ok=True)
    print(f"[info] Saving weights to : {weight_save_dir}")

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Validation loss", mode="min"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)
    print(f"[info] Using dataloader: {dict_args['dataloader']}")

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        val_check_interval=dict_args['val_freq'],
        deterministic=True,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        resume_from_checkpoint=dict_args['resume_from_checkpoint']
    )

    # Trainer: train model
    print(f"[info] Starting training")
    trainer.fit(model, data_module)

    # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    if dict_args['eval']:
        trainer.test(model, ckpt_path='best', datamodule=data_module)
    else:
        print("Evaluation skipped")
