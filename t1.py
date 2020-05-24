# a = {}
# for i in range(5):
#     if 'haha' not in a.keys():
#         a['haha'] = []
#     a['haha'].append(i)
# print(a)
# a = [1, 2, 3, 4, 5]
# b,c,d = a[:3]
# print(b, c, d)
# import tensorflow as tf
# a = tf.placeholder(name='a',shape=[5,12,13],dtype=tf.float32)
# lstm_1 = tf.keras.layers.CuDNNLSTM(units=50, return_sequences=True, return_state=True)
# lstm_2 = tf.keras.layers.CuDNNLSTM(units=50, return_sequences=True, return_state=True)
# lstm_3 = tf.keras.layers.CuDNNLSTM(units=50, return_sequences=False, return_state=True)
# b = lstm_1(a)
# b = lstm_2(b[0])
# b = lstm_3(b[0])
# print(b[0])
# e = tf.placeholder(name='e',shape=[5,13],dtype=tf.float32)
# b = tf.layers.conv1d(a, filters=10, kernel_size=2, padding='SAME')
# c = tf.random_normal(shape=a.get_shape().as_list(), mean=0.0, stddev=1.0)
# e = tf.expand_dims(e, axis=1)
# f = tf.stack((a, e), axis=1)
# print(f)
# print(c)
# print(b)
# import tensorflow as tf
# a = tf.placeholder(name='a',shape=[5,None,13],dtype=tf.float32)
# b = tf.random_normal(shape=tf.shape(a), mean=0.0, stddev=1.0)
# print(b)
# from utils import get_sp
# sp = get_sp('./data/wavs/E10001.wav')
# print(sp.shape)
import tensorflow as tf
a = tf.placeholder(name='a', shape=[5, 10, 12], dtype=tf.float32)
b = tf.layers.conv1d(inputs=a, filters=1, kernel_size=5, padding='VALID')
print(b)
