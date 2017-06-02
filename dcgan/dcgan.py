import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


class G_conv():
    def __init__(self, meta_data):
        self._a0 = int(meta_data['img_a'])
        self._a1 = int(self._a0 / 2)
        self._a2 = int(self._a1 / 2)
        self._a3 = int(self._a2 / 2)
        self._a4 = int(self._a3 / 2)
        self._s4 = self._a4 * self._a4

    def __call__(self, z, reuse=False):
        with tf.variable_scope('gen', reuse=reuse):
            # project z into conv context
            r = layers.fully_connected(
                z, num_outputs=self._s4*1024,
                activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm
            )
            r = tf.reshape(r, [-1, self._a4, self._a4, 1024])
            # transpose conv2d
            r = layers.conv2d_transpose(
                r, 512, 3, stride=2,
            )

class D_conv():
    def __init__(self):
        pass

    def __call__(self, x, reuse=False):
        with tf.variable_scope('dis', reuse=reuse):
            pass


class DCGAN():
    # meta_data:
    # 'z_dim': dims for the noise z
    # 'img_a': img_a x img_a is the img size
    # 'img_c': the channels of img
    def __init__(self, G, D, meta_data):
        self._G = G(meta_data)
        self._D = D(meta_data)

        # set meta data
        self._z_dim = meta_data['z_dim']
        self._img_a = meta_data['img_a']
        self._img_c = meta_data['img_c']

        # generate sample
        self._z = tf.placeholder(tf.float32, [None, self._z_dim])
        self._x = tf.placeholder(
            tf.float32, [None, self._img_a, self._img_a, self._img_c])
        self._true_x = self._x
        self._fake_x = self._G(self._z)

        # discriminate
        self._true_logits = self._D(self._true_x)
        self._fake_logits = self._D(self._fake_x, reuse=True)

        # loss for D
        self._fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self._fake_logits),
                logits=self._fake_logits
            )
        )
        self._true_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._true_logtis),
                logits=self._true_logits
            )
        )
        self._D_loss = self._fake_loss + self._true_loss

        # loss for G
        self._G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._fake_logits),
                logits=self._fake_logits
            )
        )

        self._saver = tf.train.Saver()

    def train(
                self, sess, G_optim, D_optim, data_set, 
                mb_size=32, epoches=1000000, D_K=1, G_K=1):
        G_optim = G_optim.minize(self._G_loss, var_list=self._G.theta)
        D_optim = D_optim.minize(self._D_loss, var_list=self._D.theta)
        data_set.reset()
        for epoch in range(epoches):
            # 1. sample
            # 2. update
            x_b = data_set.next_batch(mb_size)
            # update discriminator
            D_loss = 0
            for _ in range(D_K):
                loss, _ = sess.run(
                    [self._D_loss, D_optim],
                    feed_dict={
                        self._x: x_b,
                        self._z: sample_z(mb_size, self._z_dim)
                    }
                )
                D_loss += loss
            D_loss /= D_K
            # update generator
            G_loss = 0
            for _ in range(G_K):
                loss, _ = sess.run(
                    [self._G_loss, G_optim],
                    feed_dict={self._z: sample_z(mb_size, self._z_dim)}
                )
                G_loss += loss
            G_loss /= G_K
