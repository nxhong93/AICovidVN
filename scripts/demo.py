import __init__
import numpy as np
import pandas as pd
import argparse
import os
from joblib import Parallel, delayed
from config.configs import *
from scripts.utils import *
from scripts.engineer import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import AUROC
from pathlib import Path


parser = argparse.ArgumentParser(description='Demo Config', add_help=False)
parser.add_argument('--list_file', default='./demo_data')
demo_args = parser.parse_args()

if __name__ == '__main__':
    train_df = pd.read_csv(args.train_csv)
    list_file = [i.split('.')[0] for i in os.listdir(demo_args.list_file)]
    list_path = [os.path.join(demo_args.list_file, i) for i in os.listdir(demo_args.list_file)]
    demo_df = pd.DataFrame({'uuid': list_file, 'file_path': list_path})
    demo_df['start'] = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(startClean)(path) for path in demo_df.file_path)
    demo_df['end'] = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(endClean)(path) for path in demo_df.file_path)

    model = covidNet.load_from_checkpoint(checkpoint_path=args.checkpoint, config=TrainConfig,
                                          meta_col=meta_cols, df=train_df, test_df=demo_df,
                                          fold=0, is_train=False, oof=False, is_sub=False)
    trainer = pl.Trainer(gpus=[0], auto_scale_batch_size='binsearch', num_sanity_val_steps=0)
    trainer.test(model)
    demo_df['predict'] = model.pred
    pred_dict = {id_: f'{pred:.4f}' for (id_, pred) in demo_df[['uuid', 'predict']].values}

    print(pred_dict)
