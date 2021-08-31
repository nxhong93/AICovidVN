import __init__
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from config.configs import *


def train_process(sub='train'):
    if sub == 'train':
        kf = KFold(n_splits=FOLD, random_state=SEED, shuffle=True)

        train_df = pd.read_csv(args.train_origin_csv)
        train_df['file_path'] = train_df['uuid'].apply(lambda x: os.path.join(args.train_origin, x + '.wav'))
        train_df['start'] = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(startClean)(path) for path in tqdm(train_df.file_path, total=len(train_df)))
        train_df['end'] = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(endClean)(path) for path in tqdm(train_df.file_path, total=len(train_df)))
        if 'subject_gender' in train_df.columns and 'subject_age' in train_df.columns:
            train_df['gender'] = train_df['subject_gender'].apply(lambda x: gender[x])
            age_df = pd.get_dummies(train_df['subject_age'])
            train_df = pd.concat([train_df, age_df], axis=1).reset_index(drop=True)
            for fold, (train_idx, valid_idx) in enumerate(
                    kf.split(train_df, train_df.assessment_result, train_df[['subject_age', 'subject_gender']])):
                train_df.loc[valid_idx, 'fold'] = fold
        else:
            for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, train_df.assessment_result)):
                train_df.loc[valid_idx, 'fold'] = fold
        train_df['fold'] = train_df['fold'].astype(int)
        train_df.to_csv(args.train_csv, index=False)
    elif sub == 'test':
        list_file = [i.split('.')[0] for i in os.listdir(args.test_origin)]
        list_path = [os.path.join(args.test_origin, i) for i in os.listdir(args.test_origin)]
        test_df = pd.DataFrame({'uuid': list_file, 'file_path': list_path})
        test_df['start'] = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(startClean)(path) for path in tqdm(test_df.file_path, total=len(test_df)))
        test_df['end'] = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(endClean)(path) for path in tqdm(test_df.file_path, total=len(test_df)))
        test_df.to_csv(args.test_csv, index=False)


if __name__ == '__main__':
    train_process(sub='train')
    train_process(sub='test')
