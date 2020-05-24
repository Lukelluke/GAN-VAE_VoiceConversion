import tensorflow as tf
from hyperparams import hyperparams
from modules import get_next_batch
from utils import learning_rate_decay, control_weight
from modules import speaker_embedding
import numpy as np
from modules import generator, discriminator
hp = hyperparams()

class Graph:
    def __init__(self, mode='train'):
        self.mode = mode
        self.model_name = 'vae_gan_vc'
        self.reuse = tf.AUTO_REUSE
        if self.mode in ['train']:
            self.is_training = True
            with tf.device('/gpu:0'):
                self.train()
            tf.summary.scalar('{}/kl_loss_weight'.format(self.mode), self.kl_loss_weight)
            tf.summary.scalar('{}/kl_loss'.format(self.mode), self.kl_loss)
            tf.summary.scalar('{}/reconstruction_loss'.format(self.mode), self.reconstruction_loss)
            tf.summary.scalar('{}/cycle_loss'.format(self.mode), self.cycle_loss)
            tf.summary.scalar('{}/GAN_G_loss'.format(self.mode), self.GAN_G_loss)
            tf.summary.scalar('{}/G_loss'.format(self.mode), self.G_loss)
            tf.summary.scalar('{}/D_loss'.format(self.mode), self.D_loss)
            self.merged = tf.summary.merge_all()
            self.t_vars = tf.trainable_variables()
            self.num_paras = 0
            for var in self.t_vars:
                var_shape = var.get_shape().as_list()
                self.num_paras += np.prod(var_shape)
            print('Total number of trainable parameters : %r' % self.num_paras)
        elif self.mode in ['test']:
            self.is_training = False
            with tf.device('/cpu:0'):
                self.test()
        elif self.mode in ['infer']:
            self.is_training = False
            with tf.device('/cpu:0'):
                self.inference()
        else:
            raise Exception('No supported mode in model __init__ function. Please check.')

    def flow(self):
        self.ori_embed = speaker_embedding(self.ori_spk, hp.SPK_NUM, hp.EMBED_SIZE, reuse=False) # [N, 1] -> [N, E]
        self.aim_embed = speaker_embedding(self.aim_spk, hp.SPK_NUM, hp.EMBED_SIZE, reuse=True) # [N, 1] -> [N, E]
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.ori_out, self.ori_mu, self.ori_log_var = generator(self.ori_embed,
                                                                    self.ori_feat,
                                                                    is_training=self.is_training,
                                                                    reuse=False,
                                                                    scope_name='generator')
            self.aim_out, self.aim_mu, self.aim_log_var = generator(self.aim_embed,
                                                                    self.ori_feat,
                                                                    is_training=self.is_training,
                                                                    reuse=True, scope_name='generator')
            self.cycle_ori_out, self.cycle_mu, self.cycle_log_var = generator(self.ori_embed,
                                                                              self.aim_out,
                                                                              is_training=self.is_training,
                                                                              reuse=True,
                                                                              scope_name='generator')
            self.predict_real_P = discriminator(self.aim_feat,
                                                reuse=False,
                                                scope_name='discriminator')
            self.predict_fake_P = discriminator(self.aim_out,
                                                reuse=True,
                                                scope_name='discriminator')

    def train(self):
        self.ori_spk, self.ori_feat, self.aim_spk, self.aim_feat, \
        self.t_G, self.t_D_fake, self.t_D_real = get_next_batch()
        self.flow()
        self.update()

    def update(self):
        self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
        self.generator_lr = learning_rate_decay(hp.G_LR, global_step=self.global_step)
        self.discriminator_lr = learning_rate_decay(hp.D_LR, global_step=self.global_step)

        # Generator loss
        self.reconstruction_loss = tf.reduce_mean(tf.abs(self.ori_out - self.ori_feat))
        self.cycle_loss = tf.reduce_mean(tf.abs(self.cycle_ori_out - self.ori_feat))
        self.ori_kl_loss = - 0.5 * tf.reduce_sum(1 + self.ori_log_var - tf.pow(self.ori_mu, 2) - tf.exp(self.ori_log_var))
        self.aim_kl_loss = - 0.5 * tf.reduce_sum(1 + self.aim_log_var - tf.pow(self.aim_mu, 2) - tf.exp(self.aim_log_var))
        self.cycle_kl_loss = - 0.5 * tf.reduce_sum(1 + self.cycle_log_var - tf.pow(self.cycle_mu, 2) - tf.exp(self.cycle_log_var))
        self.kl_loss_weight = control_weight(self.global_step)
        self.kl_loss = self.kl_loss_weight * (self.ori_kl_loss + self.aim_kl_loss + self.cycle_kl_loss)
        self.GAN_G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t_G, logits=self.predict_fake_P))
        self.G_loss = self.reconstruction_loss + self.cycle_loss + self.kl_loss + self.GAN_G_loss

        # Discriminator loss
        self.D_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t_D_fake, logits=self.predict_fake_P))
        self.D_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t_D_real, logits=self.predict_fake_P))
        self.GAN_D_loss = self.D_fake_loss + self.D_real_loss
        self.D_loss = self.GAN_D_loss

        # Variables
        trainable_variables = tf.trainable_variables()
        self.G_vars = [var for var in trainable_variables if 'generator' in var.name]
        self.D_vars = [var for var in trainable_variables if 'discriminator' in var.name]

        # Optimizer
        self.G_optimizer = tf.train.AdamOptimizer(self.generator_lr)
        self.D_optimizer = tf.train.AdamOptimizer(self.discriminator_lr)

        # Generator Gradient Clipping And Update
        self.G_clipped = []
        self.G_gvs = self.G_optimizer.compute_gradients(self.G_loss, var_list=self.G_vars)
        for grad, var in self.G_gvs:
            grad = tf.clip_by_norm(grad, 5.)
            self.G_clipped.append((grad, var))
        self.G_train_op = self.G_optimizer.apply_gradients(self.G_clipped, global_step=self.global_step)

        # Discriminator Gradient Clipping And Update
        self.D_clipped = []
        self.D_gvs = self.D_optimizer.compute_gradients(self.D_loss, var_list=self.D_vars)
        for grad, var in self.D_gvs:
            grad = tf.clip_by_norm(grad, 5.)
            self.D_clipped.append((grad, var))
        self.D_train_op = self.D_optimizer.apply_gradients(self.D_clipped, global_step=self.global_step)

    def test(self):
        pass

    def inference(self):
        self.ori_feat = tf.placeholder(name='ori_feat', shape=[None, None, hp.CODED_DIM], dtype=tf.float32)
        self.aim_spk = tf.placeholder(name='aim_spk', shape=[None, None], dtype=tf.int64)
        #self.flow()
        self.aim_embed = speaker_embedding(self.aim_spk, hp.SPK_NUM, hp.EMBED_SIZE) # [N, 1] -> [N, E]
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.aim_out, self.aim_mu, self.aim_log_var = generator(self.aim_embed,
                                                                    self.ori_feat,
                                                                    is_training=self.is_training,
                                                                    scope_name='generator')
