import __init__
import numpy as np
import pandas as pd
from config.configs import *
import dataset
import network
from engineer import covidNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import AUROC
from pathlib import Path


if __name__ == '__main__':
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    print(test_df)
    print(100 * '-')

    model = covidNet.load_from_checkpoint(checkpoint_path=args.checkpoint, config=TrainConfig,
                                          meta_col=meta_cols, df=train_df, test_df=test_df,
                                          fold=0, is_train=False, oof=False)
    trainer = pl.Trainer(gpus=[0], auto_scale_batch_size='binsearch', num_sanity_val_steps=0)
    trainer.test(model)
