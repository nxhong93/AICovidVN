# import __init__
from config.configs import *
import librosa
import numpy as np


def startClean(path, window=10):
    y = librosa.load(path, sr=SR)[0]
    threshold = 1 / 2 * np.max(y)
    for i in range(len(y)):
        if y[i:min(i + window, len(y))].sum() >= threshold:
            return i
    return 0


def endClean(path, window=10):
    y = librosa.load(path, sr=SR)[0]
    threshold = 1 / 2 * np.max(y)
    for i in range(len(y), 0, -1):
        if y[max(i - window, 0):i].sum() >= threshold:
            return i
    return len(y)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V
