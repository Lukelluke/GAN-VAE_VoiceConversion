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

"""
合成要sp和f0
sp是网络预测的

找好以下内容：
原始的sp，ap，f0，编码过的sp
目标人的sp，ap，f0，编码过的sp
"""

def synthesis(ori_path, aim_sp, aim_spkid):
    print('synthesizing ...')
    wav, _ = librosa.load(ori_path, sr=hp.SR, mono=True, dtype=np.float64)
    f0, timeaxis = pw.harvest(wav, hp.SR)
    sp_per_timeaxis_before = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT) # 1024 压缩到 513 维


    # ori_decoded_sp = pw.decode_spectral_envelope(ori_sp, hp.SR, fft_size=hp.N_FFT)

    # print('f0.shape = ')
    # print(f0)

    ap = pw.d4c(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    aim_decoded_sp = pw.decode_spectral_envelope(aim_sp, hp.SR, fft_size=hp.N_FFT)  # 转换/解码 后的sp：
    print('解码后的513维度的aim_decoded_sp = ')
    print(aim_decoded_sp.shape)
    print(aim_decoded_sp[399][:])


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
    for spk,num in  fpath_aim.items():
        if num == aim_spkid:
            print(spk)
            aim_spkname = aim_spkname + spk  # 字符串拼接，成为路径

    files = os.listdir(aim_spkname)  # 存储目标人文件的文件夹
    aimpath = os.path.join(aim_spkname, files[0])
    print(aimpath)  # ./data/convert/TMM1/M10007.wav



    source_mean_f0 = get_mean_f0(fpath)
    target_mean_f0 = get_mean_f0(aimpath)
    diff_mean_f0 = target_mean_f0 - source_mean_f0
    source_mean_f0 += diff_mean_f0
    print(source_mean_f0)  # 这里的source_mean_f0还只是一个数字




    ori_feat = get_sp(fpath)  # 斌的：原始说话人的sp，接下来用np.random.normal生成同样形状的
    print('源说话人的sp：')
    print(ori_feat)  # sp 是二维的！！！
    # 合成要用目标说话人的sp啊！不然帧数对不上
    # mine:不对，应该要用目标人的预测出来的 sp，【60*628】，直接取.shape[0]=628就可以
    # aim_feat = get_sp(aimpath)  # 目标说话人的sp，为了用它的形状
    # aim_feat_to_onedim = aim_feat.flatten()

    # print(aim_feat.shape)    # sp.shape = [60 * 531]=维度 * 帧
    # sp维度 帧*维度，f0 维度 帧

    #  换到下面预测的地方去：
    #  ori_new_f0 = np.random.normal(source_mean_f0,1.0,aim_feat.shape[0])  # 以目标说话人的形状塑造出来的新f0
    # 维度不对，f0是一维，sp是二维
    #print(ori_new_f0) # 这是得到的新f0
    # print(ori_feat)
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
            ori_feat_batch = np.expand_dims(ori_feat, axis=0)   # 模型训练的时候是三维度的
            aim_spkid_batch = np.array([[aim_spkid]])
            for j in tqdm.tqdm(range(1)):
                aim_out = sess.run(G.aim_out, {G.ori_feat: ori_feat_batch, G.aim_spk: aim_spkid_batch})

            print('\n ori_feat_batch.shape = ')
            print(ori_feat_batch.shape)
            print('aim_spkid_batch,shape = ')
            print(aim_spkid_batch.shape)


            aim_out = np.array(aim_out, dtype=np.float64)
            predict_sp = np.reshape(aim_out, (-1, hp.CODED_DIM))
            # print(predict_sp.shape)  # [628,60]:sp维度 帧*维度，f0 维度 帧
            # print('predict_sp.shape[0]=')
            # print(predict_sp.shape[0])  # 628
            ori_new_f0 = np.random.normal(source_mean_f0, 1.0, predict_sp.shape[0])

            print('源说话人的 400 帧的60维的 sp = ')
            print(ori_feat_batch[0][399][:])  # (B, T, CODED_DIMS)
            print(aim_spkid_batch[0][0])

            print('predict_sp.shape = ')
            print(predict_sp.shape)
            print('\n目标说话人的 400 帧的60维的 sp = ')
            print(predict_sp[399][:])  # (B, T, CODED_DIMS)

            for i in range(len(ori_feat_batch[0][399][:])):
                print(str(i)+',' +str(ori_feat_batch[0][399][i])+',' + str(predict_sp[399][i]))

            print('Sp predicted done.')
            synthesis(fpath, predict_sp, aim_spkid)

if __name__ == '__main__':
    main()
