import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--train_csv', default='./data/train_clean.csv')
parser.add_argument('--test_csv', default='./data/test_clean.csv')
parser.add_argument('--train_warm_path',
                    default='./data/aicv115m/aicv115m_public_train/train_audio_files_8k/train_audio_files_8k/')
parser.add_argument('--train_file_path',
                    default='./data/aicv115m/aicv115m_final_public_train/aicv115m_final_public_train/public_train_audio_files/')
parser.add_argument('--test_file_path',
                    default='./data/aicv115m/aicv115m_final_private_test/aicv115m_final_private_test/private_test_audio_files')
parser.add_argument('--save_path', default='./weights')
parser.add_argument('--checkpoint', default='./weights/epoch=36-step=16760.ckpt')
args = parser.parse_args()

SR = 32000
FOLD = 5
list_age = ['group_0_2', 'group_3_5', 'group_6_13', 'group_14_18', 'group_19_33',
            'group_34_48', 'group_49_64', 'group_65_78', 'group_79_98']
meta_cols = ['gender', 'group_0_2', 'group_3_5', 'group_6_13', 'group_14_18',
             'group_19_33', 'group_34_48', 'group_49_64', 'group_65_78', 'group_79_98']


class DatasetConfig:
    sr = SR
    period = 10
    max_length = 384
    threshold = 0.1
    melspectrogram_config = {'sr': SR,
                             'verbose': False,
                             'hop_length': 256,
                             'n_fft': 1024,
                             'n_mels': 384,
                             'fmin': 20,
                             'fmax': 20000}
    mfcc_config = {'sr': SR,
                   'verbose': False,
                   'n_mfcc': 256}


class TrainConfig:
    n_epochs = 50
    model_name = 'tf_efficientnet_b0_ns'
    lr = 3e-5
    optimizer = 'madgrad'
    useMeta = False
    momentum = 0.9
    SchedulerCosine = CosineAnnealingLR
    cosine_params = dict(
        T_max=n_epochs,
        eta_min=1e-7,
        verbose=True
    )
    has_warmup = True
    warmup_params = {
        'multiplier': 5,
        'total_epoch': 1,
    }
    alpha = None
    use_focal = False
    use_gem = False
    use_weight = False
    batch_size = 8
    num_workers = 12
    num_tta = 1
