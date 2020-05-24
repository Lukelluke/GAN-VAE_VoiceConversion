import tensorflow as tf
import glob
from hyperparams import hyperparams
from networks import encoder, decoder, conv1d, bn, prenet, gru
hp = hyperparams()
def get_next_batch():
    tfrecords = glob.glob(f'{hp.TRAIN_DATASET_PATH}/*.tfrecord')
    filename_queue = tf.train.string_input_producer(tfrecords, shuffle=True, num_epochs=hp.NUM_EPOCHS)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'ori_spkid': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'ori_feat': tf.VarLenFeature(dtype=tf.float32),
            'ori_feat_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),

            'aim_spkid': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'aim_feat': tf.VarLenFeature(dtype=tf.float32),
            'aim_feat_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),

            'target_G': tf.FixedLenFeature(shape=(hp.SPK_NUM*2,), dtype=tf.float32),
            'target_D_fake': tf.FixedLenFeature(shape=(hp.SPK_NUM*2,), dtype=tf.float32),
            'target_D_real': tf.FixedLenFeature(shape=(hp.SPK_NUM*2,), dtype=tf.float32)
        }
    )
    features['ori_feat'] = tf.sparse_tensor_to_dense(features['ori_feat'])
    features['aim_feat'] = tf.sparse_tensor_to_dense(features['aim_feat'])
    ori_spk = features['ori_spkid']
    ori_feat = tf.reshape(features['ori_feat'], features['ori_feat_shape'])
    aim_spk = features['aim_spkid']
    aim_feat = tf.reshape(features['aim_feat'], features['aim_feat_shape'])
    target_G = features['target_G']
    target_D_fake = features['target_D_fake']
    target_D_real = features['target_D_real']
    ori_feat = tf.reshape(ori_feat, [-1, hp.CODED_DIM])
    aim_feat = tf.reshape(aim_feat, [-1, hp.CODED_DIM])
    ori_spk_batch, ori_feat_batch, aim_spk_batch, aim_feat_batch, \
    target_G_batch, target_D_fake_batch, target_D_real_batch = tf.train.batch([ori_spk, ori_feat, aim_spk, aim_feat,
                                                                               target_G, target_D_fake, target_D_real],
                                                                              batch_size=hp.BATCH_SIZE,
                                                                              capacity=100,
                                                                              num_threads=10,
                                                                              dynamic_pad=True,
                                                                              allow_smaller_final_batch=False)
    return ori_spk_batch, ori_feat_batch, aim_spk_batch, aim_feat_batch,\
           target_G_batch, target_D_fake_batch, target_D_real_batch

def speaker_embedding(inputs, spk_num, num_units, zero_pad=True, scope="speaker_embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      spk_num: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[spk_num, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def generator(speaker_embedding, inputs, is_training=True, scope_name='generator', reuse=None):
    '''Generate features.

    Args:
      speaker_embedding: A `Tensor` with type `float32` contains speaker information. [N, E]
      inputs: A `Tensor` with type `float32` contains speech features.
      is_training: Boolean, whether to train or inference.
      scope_name: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A decoded `Tensor` with aim speaker.
      vae mu vector.
      vae log_var vector.
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        sample, mu, log_var = encoder(inputs, is_training=is_training, scope='vae_encoder') # [N, T, E]
        #speaker_embedding = tf.expand_dims(speaker_embedding, axis=1) # [N, 1, E]
        speaker_embedding = tf.tile(speaker_embedding, [1, tf.shape(sample)[1], 1]) # [N, T, E]
        encoded = tf.concat((speaker_embedding, sample), axis=-1) # [N, T, E+G]
        outputs = decoder(encoded, is_training=is_training, scope='vae_decoder')
        return outputs, mu, log_var # [N, T, C]

def new_generator(speaker_embedding, inputs, is_training=True, scope_name='generator', reuse=None):
    '''Generate features.

    Args:
      speaker_embedding: A `Tensor` with type `float32` contains speaker information. [N, E]
      inputs: A `Tensor` with type `float32` contains speech features.
      is_training: Boolean, whether to train or inference.
      scope_name: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A decoded `Tensor` with aim speaker.
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        # encoder
        encoder_outs_1 = conv1d(inputs, filters=hp.ENCODER_FILTER_NUMS, size=hp.ENCODER_FILTER_SIZE, scope="encoder_conv1d_1")
        encoder_outs_1 = bn(encoder_outs_1, is_training=is_training, activation_fn=tf.nn.relu, scope="encoder_conv1d_1") # (N, T, E1)
        encoder_outs_2 = conv1d(encoder_outs_1, filters=hp.ENCODER_FILTER_NUMS//2, size=hp.ENCODER_FILTER_SIZE, scope="encoder_conv1d_2")
        encoder_outs_2 = bn(encoder_outs_2, is_training=is_training, activation_fn=tf.nn.relu, scope="encoder_conv1d_2") # (N, T, E1//2)
        z = gru(encoder_outs_2, num_units=hp.ENCODER_GRU_UNITS, bidirection=False, scope="encoder_gru_1") # (N, T, U1)
        speaker_embedding = tf.tile(speaker_embedding, [1, tf.shape(z)[1], 1]) # (N, T, E)
        condional_z = tf.concat((speaker_embedding, z), axis=-1) # (N, T, E + U1)
        # decoder
        decoder_inputs = tf.concat((encoder_outs_2, condional_z), axis=-1) # (N, T, E1//2 + E + U1)
        decoder_outs_1 = conv1d(decoder_inputs, filters=hp.DECODER_FILTER_NUMS, size=hp.DECODER_FILTER_SIZE, scope="decoder_conv1d_1")
        decoder_outs_1 = bn(decoder_outs_1, is_training=is_training, activation_fn=tf.nn.relu, scope='decoder_conv1d_1') # (N, T, E2)
        decoder_inputs_2 = tf.concat((decoder_outs_1, encoder_outs_2), axis=-1) # [N, T, E1//2 + E2]
        decoder_outs_2 = conv1d(decoder_inputs_2, filters=hp.DECODER_FILTER_NUMS*2, size=hp.DECODER_FILTER_SIZE, scope='decider_conv1d_2')
        decoder_outs_2 = bn(decoder_outs_2, is_training=is_training, scope='decoder_conv1d_2') # [N, T, E2*2]
        decoder_inputs_3 = tf.concat((decoder_outs_2, encoder_outs_1), axis=-1) # [N, T, E2*2 + E1]
        outs = gru(decoder_inputs_3, num_units=hp.DECODER_GRU_UNITS, bidirection=False, scope='decoder_gru_1') # (N, T, U2)
        outs = tf.layers.dense(outs, units=hp.CODED_DIM, activation=tf.nn.relu, name='decoder_dense_1') # (N, T, D)
        return outs




def discriminator(inputs, scope_name='discriminator', reuse=None):
    '''Discriminator features.

    Args:
      inputs: A `Tensor` with type `float32` contains speech features. [N, T, F]
      scope_name: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A softmax
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        out = lstm_3_layers(inputs, num_units=hp.LSTM_UNITS, bidirection=False) # [N, U]
        out = tf.layers.dense(out, units=hp.LSTM_UNITS//2, activation=tf.nn.tanh, name='dense1') # [N, U//2]
        out = tf.layers.dense(out, units=hp.SPK_NUM*2, activation=tf.nn.sigmoid, name='dense2') # [N, L]
        return out

def fast_lstm_3_layers(inputs, num_units=None, bidirection=False, scope="lstm_3_layers", reuse=tf.AUTO_REUSE):
    '''
    :param inputs: A 3-d tensor. [N, T, C]
    :param num_units: An integer. The last hidden units.
    :param bidirection: A boolean. If True, bidirectional results are concatenated.
    :param scope: A string. scope name.
    :param reuse: Boolean. whether to reuse the weights of a previous layer.
    :return: if bidirection is True, A 2-d tensor. [N, num_units * 2]
             else, A 2-d tensor. [N, num_units]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if not num_units:
            num_units = inputs.get_shape().as_list[-1]
        with tf.variable_scope('lstm_1'):
            lstm_1 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=True, return_state=True)
        with tf.variable_scope('lstm_2'):
            lstm_2 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=True, return_state=True)
        with tf.variable_scope('lstm_3'):
            lstm_3 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=False, return_state=True)
        out = lstm_1(inputs)
        out = lstm_2(out[0])
        out = lstm_3(out[0])
        return out[0]

def lstm_3_layers(inputs, num_units=None, bidirection=False, scope="lstm", reuse=tf.AUTO_REUSE):
    '''
    :param inputs: A 3-d tensor. [N, T, C]
    :param num_units: An integer. The last hidden units.
    :param bidirection: A boolean. If True, bidirectional results are concatenated.
    :param scope: A string. scope name.
    :param reuse: Boolean. whether to reuse the weights of a previous layer.
    :return: if bidirection is True, A 2-d tensor. [N, num_units * 2]
             else, A 2-d tensor. [N, num_units]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if not num_units:
            num_units = inputs.get_shape().as_list[-1]
        # cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
        cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cellls)
        if bidirection:
            bw_cells = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
            multi_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
            outputs, final_state = tf.nn.dynamic_rnn(multi_cell, multi_bw_cell, inputs=inputs, dtype=tf.float32)
            # outputs shape : top lstm outputs, ([N, T, num_units], [N, T, num_units])
            # lstm final_state : multi final state stack together, ([N, 2, num_units], [N, 2, num_units])
            return tf.concat(final_state, axis=2)[-1][0]
        outputs, final_state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=inputs, dtype=tf.float32)
        # outputs shape : top lstm outputs, [N, T, num_units]
        # lstm final_state : multi final state stack together, [N, 2, num_units]
        return final_state[-1][0]
