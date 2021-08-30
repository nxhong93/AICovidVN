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
    for fold_idx in range(FOLD-4):
        print(100 * '-')
        print(f'Fold{fold_idx}: ')

        model = covidNet(config=TrainConfig, meta_col=meta_cols, df=train_df, test_df=train_df, fold=fold_idx, is_train=True, oof=True)
        checkpoint_callback = ModelCheckpoint(dirpath=f'{args.save_path}/fold{fold_idx}', save_top_k=1, verbose=True, monitor='score', mode='max')
        Path(f'{args.save_path}/fold{fold_idx}').mkdir(parents=True, exist_ok=True)
        trainer = pl.Trainer(gpus=[0], max_epochs=TrainConfig.n_epochs,
                             auto_scale_batch_size='binsearch', num_sanity_val_steps=0,
                             callbacks=checkpoint_callback, logger=False,
                             default_root_dir=f'{args.save_path}/fold{fold_idx}')
        trainer.fit(model)
