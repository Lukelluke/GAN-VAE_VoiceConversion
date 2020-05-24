from model import Graph
import tensorflow as tf
from hyperparams import hyperparams
import os
hp = hyperparams()


def main():
    mode = 'train'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=hp.GPU_RATE)
    with tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(hp.LOG_DIR, sess.graph)
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
        except:
            print('Load trained model failed, start training with initializer ...')
            sess.run(tf.global_variables_initializer())
        finally:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    steps = 1
                    G_loss = 0
                    D_loss = 0
                    if steps % 5 != 0:
                        _, reconstruction_loss, cycle_loss, GAN_G_loss, G_loss, summary, steps = sess.run([G.G_train_op,
                                                                                                           G.reconstruction_loss,
                                                                                                           G.cycle_loss,
                                                                                                           G.GAN_G_loss,
                                                                                                           G.G_loss,
                                                                                                           G.merged,
                                                                                                           G.global_step])
                        print('train mode \t steps : {} \t '
                              'reconstruction_loss : {} \t '
                              'cycle_loss : {} \t '
                              'GAN_G_loss : {} \t '
                              'G_total_loss : {}'.format(steps,
                                                         reconstruction_loss,
                                                         cycle_loss,
                                                         GAN_G_loss,
                                                         G_loss))
                    else:
                        _, D_fake_loss, D_real_loss, D_loss, summary, steps = sess.run([G.D_train_op,
                                                                                        G.D_fake_loss,
                                                                                        G.D_real_loss,
                                                                                        G.D_loss,
                                                                                        G.merged,
                                                                                        G.global_step])
                        print('train mode \t steps : {} \t '
                              'D_fake_loss : {} \t '
                              'D_real_loss : {} \t '
                              'D_total_loss : {}'.format(steps,
                                                         D_fake_loss,
                                                         D_real_loss,
                                                         D_loss))
                    writer.add_summary(summary=summary, global_step=steps)
                    if steps % (hp.PER_STEPS + 1) == 0:
                        saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_%.3fGlos_%.3fDlos_%dsteps' % (G_loss,
                                                                                                         D_loss,
                                                                                                         steps)))

            except tf.errors.OutOfRangeError:
                print('Training Done.')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
