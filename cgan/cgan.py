import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_y(bs):
    r = []
    for i in range(bs):
        idx = i % 10
        t = np.zeros((10))
        t[idx] = 1
        r.append(t)
    return np.asarray(r, dtype=np.float32)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def concat(z, y):
    return tf.concat([z, y], axis=1)


def concat_conv(x, y):
    x_shape = tf.shape(x)
    y_dim = int(y.shape[1])
    bs = x_shape[0]
    y = tf.reshape(y, shape=[bs, 1, 1, y_dim])
    x_like_y = y * tf.ones(
        [bs, x_shape[1], x_shape[2], y_dim]
    )
    return tf.concat([x, y * x_like_y], axis=3)


class G_conv():
    def __init__(self, meta_data):
        self._a0 = int(meta_data['img_a'])  # 28
        self._a1 = int(self._a0 / 2)        # 14
        self._a2 = int(self._a1 / 2)        # 7
        self._s2 = self._a2 * self._a2
        self._ch = int(meta_data['img_c'])
        self._w_init = tf.random_normal_initializer(0, 0.02)
        self._scope = 'gen'

    def __call__(self, z, reuse=False):
        with tf.variable_scope(self._scope, reuse=reuse):
            # project z into conv context
            r = layers.fully_connected(
                z, num_outputs=self._s2*128,
                activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm
            )
            r = tf.reshape(r, [-1, self._a2, self._a2, 128])
            # transpose conv2d
            r = layers.conv2d_transpose(
                r, 64, 4, stride=2,
                activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm,
                padding='SAME', weights_initializer=self._w_init
            )
            r = layers.conv2d_transpose(
                r, self._ch, 4, stride=2,
                activation_fn=tf.nn.sigmoid,
                padding='SAME', weights_initializer=self._w_init
            )
            return r

    @property
    def theta(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self._scope
        )


class D_conv():
    def __init__(self, meta_data):
        self._scope = 'dis'
        self._a = meta_data['img_a']
        self._w_init = tf.random_normal_initializer(0, 0.02)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self._scope, reuse=reuse):
            d = layers.conv2d(
                x, num_outputs=64, kernel_size=4,
                stride=2, activation_fn=lrelu
            )
            d = layers.conv2d(
                d, num_outputs=128, kernel_size=4,
                stride=2, activation_fn=lrelu,
                normalizer_fn=layers.batch_norm
            )
            d = layers.flatten(d)

            d = layers.fully_connected(
                d, 1, activation_fn=None,
                weights_initializer=self._w_init
            )
            return d

    @property
    def theta(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self._scope
        )


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
        self._y_dim = meta_data['y_dim']
        self._img_a = meta_data['img_a']
        self._img_c = meta_data['img_c']

        # generate sample
        self._z = tf.placeholder(tf.float32, [None, self._z_dim])
        self._y = tf.placeholder(tf.float32, [None, self._y_dim])
        self._x = tf.placeholder(
            tf.float32, [None, self._img_a, self._img_a, self._img_c])
        self._true_x = self._x
        self._fake_x = self._G(concat(self._z, self._y))

        # discriminate
        self._true_logits = self._D(concat_conv(self._true_x, self._y))
        self._fake_logits = self._D(
            concat_conv(self._fake_x, self._y),
            reuse=True
        )

        # loss for D
        self._fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self._fake_logits),
                logits=self._fake_logits
            )
        )
        self._true_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._true_logits),
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
                mb_size=128, epoches=1000000, D_K=1, G_K=1):
        G_optim = G_optim.minimize(self._G_loss, var_list=self._G.theta)
        D_optim = D_optim.minimize(self._D_loss, var_list=self._D.theta)
        sess.run(tf.global_variables_initializer())
        i = 0
        if not os.path.exists('out'):
            os.mkdir('out')
        for epoch in range(epoches):
            # 1. sample
            if epoch % 100 == 0:
                samples = sess.run(
                    self._fake_x,
                    feed_dict={
                        self._y: sample_y(16),
                        self._z: sample_z(16, self._z_dim)
                    }
                )

                fig = plot(samples)
                plt.savefig(
                    'out/{}.png'.format(str(i).zfill(3)),
                    bbox_inches='tight')
                plt.close(fig)

                img = cv2.imread('out/{}.png'.format(str(i).zfill(3)))
                cv2.imshow('sample', img)
                cv2.waitKey(1)
            if epoch % 1000 == 0:
                i += 1
            # 2. update
            x_b, y_b = data_set.next_batch(mb_size)
            x_b = np.reshape(
                x_b,
                [mb_size, self._img_a, self._img_a, self._img_c]
            )
            # update discriminator
            D_loss = 0
            for _ in range(D_K):
                loss, _ = sess.run(
                    [self._D_loss, D_optim],
                    feed_dict={
                        self._x: x_b,
                        self._y: y_b,
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
                    feed_dict={
                        self._y: y_b,
                        self._z: sample_z(mb_size, self._z_dim)
                    }
                )
                G_loss += loss
            G_loss /= G_K

            # 3. output loss info
            if epoch % 10 == 0:
                info = 'Epoch ' + str(epoch) + ': '
                info += 'G_loss %.4f ' % G_loss
                info += 'D_loss %.4f \r' % D_loss
                sys.stdout.write(info)
                sys.stdout.flush()


if __name__ == '__main__':
    meta = {
        'img_a': 28,
        'img_c': 1,
        'y_dim': 10,
        'z_dim': 100
    }
    gan = DCGAN(G_conv, D_conv, meta)

    optim = tf.train.AdamOptimizer()
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    with tf.Session() as sess:
        gan.train(
            sess,
            optim, optim,
            mnist.train
        )
