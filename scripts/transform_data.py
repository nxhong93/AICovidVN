from augument import *
import albumentations as al


def aug(sub='train'):
    if sub == 'train':
        return al.Compose([
            SpeedTuning(p=0.5),
            TimeShifting(p=0.5),
            AddGaussianNoise(p=0.5),
            Gain(p=0.5),
            PolarityInversion(p=0.5),
            CutOut(p=0.01),
        ])
    elif sub == 'validation':
        return al.Compose([
            AddGaussianNoise(p=0.5),
        ])
    elif sub == 'test':
        return al.Compose([
            AddGaussianNoise(p=0.5),
        ])
