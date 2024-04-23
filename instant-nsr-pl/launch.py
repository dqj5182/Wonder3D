import argparse
import os
from datetime import datetime
import datasets
import systems
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
from utils.misc import load_config   


def main(exp_dir, runs_dir):
    config = load_config('configs/neuralangelo-ortho-wmask.yaml', cli_args=['--train', 'dataset.root_dir=../outputs/cropsize-192-cfg1.0/', 'dataset.scene=owl'])
    config.cmd_args = {'config': 'configs/neuralangelo-ortho-wmask.yaml', 'gpu': '0'}

    trial_name = datetime.now().strftime('@%Y%m%d-%H%M%S')
    exp_dir = os.path.join(exp_dir, config.name)
    config.save_dir = os.path.join(exp_dir, trial_name, 'save')
    config.ckpt_dir = os.path.join(exp_dir, trial_name, 'ckpt')
    config.code_dir = os.path.join(exp_dir, trial_name, 'code')
    config.config_dir = os.path.join(exp_dir, trial_name, 'config')
    # import pdb; pdb.set_trace()

    num_seed = 42
    pl.seed_everything(num_seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None)

    callbacks = []
    callbacks += [
        ModelCheckpoint(
            dirpath=config.ckpt_dir,
            **config.checkpoint
        ),
        LearningRateMonitor(logging_interval='step'),
        CodeSnapshotCallback(
            config.code_dir, use_version=False
        ),
        ConfigSnapshotCallback(
            config, config.config_dir, use_version=False
        ),
        CustomProgressBar(refresh_rate=1),
    ]

    # Logger
    loggers = []
    loggers += [
        TensorBoardLogger(runs_dir, name=config.name, version=trial_name),
        CSVLogger(exp_dir, name=trial_name, version='csv_logs')
    ]
    # import pdb; pdb.set_trace()
    # Fitting    
    trainer = Trainer(
        devices=1,
        # devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy='ddp_find_unused_parameters_false',
        **config.trainer
    )
    trainer.fit(system, datamodule=dm)

    # Inference
    trainer.test(system, datamodule=dm)
    print('Finished!')


if __name__ == '__main__':
    main(exp_dir='./exp', runs_dir='./runs')