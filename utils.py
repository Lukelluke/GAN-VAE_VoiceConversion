import pyworld as pw
from hyperparams import hyperparams
import tensorflow as tf
import numpy as np
import librosa
hp = hyperparams()

def get_sp(fpath: str):
    wav, _ = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)
    f0, timeaxis = pw.harvest(wav, hp.SR, frame_period=10)
    sp = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    coded_sp = pw.code_spectral_envelope(sp, hp.SR, hp.CODED_DIM)
    return np.array(coded_sp)

def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def control_weight(global_step, function_type='logistic'):
    steps = tf.cast(global_step + 1, dtype=tf.float32)
    if function_type == 'logistic':
        return 1/(1 + tf.exp(-hp.K * (steps - hp.X0)))
    elif function_type == 'linear':
        return tf.math.minimum(1, steps/hp.X0)
    else:
        raise Exception('No Supported VAE LOSS WEIGHT FUNCTION.')

def get_mean_f0(fpath:str):  # 求原始/目标人的均值，然后再拿出去做差
    wav, _ = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)  # librosa.load 返回音频信号值 & 采样率
    f0, timeaxis = pw.harvest(wav, hp.SR)  # f0是一维数组，每帧会有一个f0
    return sum(f0) / sum(timeaxis)  # 是len还是sum？
