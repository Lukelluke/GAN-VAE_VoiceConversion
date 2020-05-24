from model import Graph
import tensorflow as tf
from hyperparams import hyperparams
import numpy as np
import tqdm
from utils import get_sp
from utils import get_mean_f0
import pyworld as pw
import librosa
import os

hp = hyperparams()


def synthesis(ori_path, aim_sp, aim_spkid):
    print('synthesizing ...')
    wav, _ = librosa.load(ori_path, sr=hp.SR, mono=True, dtype=np.float64)
    f0, timeaxis = pw.harvest(wav, hp.SR, frame_period=10)
    sp_per_timeaxis_before = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # 1024 压缩到 513 维

    ap = pw.d4c(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    aim_decoded_sp = pw.decode_spectral_envelope(aim_sp, hp.SR, fft_size=hp.N_FFT)  # 转换/解码 后的sp：维度从60变成513

    print('line23: f0.shape = ' + str(f0.shape) + 'aim_decoded_sp.shape = ' + str(aim_decoded_sp.shape) +
          'ap.shape = ' + str(ap.shape))
    print('\n line26 : aim_sp.shape = '+str(aim_sp.shape))

    synwav = pw.synthesize(f0, aim_decoded_sp, ap, hp.SR)
    print(f'synthesize done. path : ./convert_to_{aim_spkid}_test1.wav')
    librosa.output.write_wav(f'./convert_to_{aim_spkid}_test1.wav', synwav, sr=hp.SR)


def main():
    # spkid是代表你想转化到哪个人  fpath是原始音频
    fpath = './data/wavs/F10001.wav'
    aim_spkid = 1
    # 这里找一下上面这个目标人的一个音频？os.listdir
    fpath_aim = {'TMM1': 1, 'TEF2': 2, 'TGM1': 3, 'TGF1': 4, 'SEF1': 5,
                 'TEF1': 6, 'TEM1': 7, 'TFM1': 8, 'TMF1': 9, 'SEM2': 10,
                 'TFF1': 11, 'SEM1': 12, 'TEM2': 13, 'SEF2': 14}
    aim_spkname = './data/convert/'
    for spk, num in fpath_aim.items():
        if num == aim_spkid:
            print(spk)
            aim_spkname = aim_spkname + spk  # 字符串拼接，成为路径

    files = os.listdir(aim_spkname)  # 存储目标人文件的文件夹
    aimpath = os.path.join(aim_spkname, files[0])
    print(aimpath)  # ./data/convert/TMM1/M10007.wav

    ori_feat = get_sp(fpath)  # 斌的：原始说话人的sp

    mode = 'infer'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
        except:
            raise Exception(f'Load trained model failed in {hp.MODEL_DIR}, please check ...')
        finally:
            ori_feat = np.reshape(ori_feat, (-1, hp.CODED_DIM))  # 斌的：原始说话人的sp
            ori_feat_batch = np.expand_dims(ori_feat, axis=0)  # 模型训练的时候是三维度的
            aim_spkid_batch = np.array([[aim_spkid]])
            for j in tqdm.tqdm(range(1)):
                aim_out = sess.run(G.aim_out, {G.ori_feat: ori_feat_batch, G.aim_spk: aim_spkid_batch})

            aim_out = np.array(aim_out, dtype=np.float64)
            predict_sp = np.reshape(aim_out, (-1, hp.CODED_DIM))

            print('Sp predicted done.')
            synthesis(fpath, predict_sp, aim_spkid)


if __name__ == '__main__':
    main()
