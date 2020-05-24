import codecs
from hyperparams import hyperparams
from utils import get_sp
import random
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os
hp = hyperparams()

def process(args):
    (tfid, dataset) = args
    writer = tf.python_io.TFRecordWriter(os.path.join(hp.TRAIN_DATASET_PATH, f'{tfid}.tfrecord'))
    for i in tqdm(dataset):
        ori_spkid, ori_fpath, aim_spkid, aim_fpath, target_G, target_D_fake, target_D_real = i[:7]
        ori_spkid = np.array(ori_spkid)
        aim_spkid = np.array(aim_spkid)
        ori_feat = get_sp(ori_fpath)
        aim_feat = get_sp(aim_fpath)
        ori_feat_shape = np.array(ori_feat.shape)
        aim_feat_shape = np.array(aim_feat.shape)
        example = tf.train.Example(features=tf.train.Features(feature={

            'ori_spkid': tf.train.Feature(int64_list=tf.train.Int64List(value=ori_spkid.reshape(-1))),
            'ori_feat': tf.train.Feature(float_list=tf.train.FloatList(value=ori_feat.reshape(-1))),
            'ori_feat_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=ori_feat_shape.reshape(-1))),

            'aim_spkid': tf.train.Feature(int64_list=tf.train.Int64List(value=aim_spkid.reshape(-1))),
            'aim_feat': tf.train.Feature(float_list=tf.train.FloatList(value=aim_feat.reshape(-1))),
            'aim_feat_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=aim_feat_shape.reshape(-1))),

            'target_G': tf.train.Feature(float_list=tf.train.FloatList(value=target_G.reshape(-1))),
            'target_D_fake': tf.train.Feature(float_list=tf.train.FloatList(value=target_D_fake.reshape(-1))),
            'target_D_real': tf.train.Feature(float_list=tf.train.FloatList(value=target_D_real.reshape(-1))),
        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def handle(dataset):
    if hp.MULTI_PROCESS:
        cpu_nums = mp.cpu_count()
        thread_nums = int(cpu_nums * hp.CPU_RATE)
        splits = [(i, dataset[i::thread_nums])
                  for i in range(thread_nums)]
        pool = mp.Pool(thread_nums)
        pool.map(process, splits)
        pool.close()
        pool.join()
    else:
        splits = (0, dataset)
        process(splits)

def main():
    lines = codecs.open(hp.CSV_PATH, 'r').readlines()
    spk2id = {}
    id2spk = {}
    id2fpath = {}
    cnt = 0
    train_dataset = []
    test_dataset = []
    for line in lines:
        spk, fname = line.strip().split('|')
        fpath = os.path.join(hp.WAVS_PATH, fname)
        if spk not in spk2id.keys():
            cnt += 1
            spk2id[spk] = cnt
            id2spk[cnt] = spk
            id2fpath[cnt] = []
        id2fpath[spk2id[spk]].append(fpath)
    if cnt != hp.SPK_NUM:
        raise Exception('Hyperparams SPK_NUM is not correct. Please check.')
    # preprocess train dataset
    for line in lines[:int(len(lines) * hp.TRAIN_RATE)]:
        spk, fname = line.strip().split('|')
        fpath = os.path.join(hp.WAVS_PATH, fname)
        aim_rand_spkid = random.randint(1, cnt)
        aim_path = id2fpath[aim_rand_spkid][random.randint(0, len(id2fpath[aim_rand_spkid]) - 1)]
        target_G = np.zeros(shape=[cnt*2])
        target_D_real = np.zeros(shape=[cnt*2])
        target_D_fake = np.zeros(shape=[cnt*2])
        target_G[aim_rand_spkid - 1 + cnt] = 1
        target_D_real[aim_rand_spkid - 1 + cnt] = 1
        target_D_fake[aim_rand_spkid - 1] = 1
        train_dataset.append([spk2id[spk], fpath, aim_rand_spkid, aim_path, target_G, target_D_fake, target_D_real])
    handle(train_dataset)
    with open('./spk2id.txt', 'w') as f:
        for key, value in spk2id.items():
            single = str(key) + ': ' + str(value)
            f.write(single)
            f.write('\n')
    # preprocess test dataset
    for line in lines[int(len(lines) * hp.TRAIN_RATE):]:
        spk, fpath = line.strip().split('|')
        aim_rand_spkid = random.randint(1, cnt)
        test_dataset.append([spk, fpath, aim_rand_spkid])
    with open('{}/test_dataset.txt'.format(hp.TEST_DATASET_PATH), 'w') as f:
        for i in test_dataset:
            f.write(i[0] + '|' + i[1] + '|' + i[2])
            f.write('\n')

if __name__ == '__main__':
    main()
