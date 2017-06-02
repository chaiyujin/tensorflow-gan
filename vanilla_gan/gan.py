import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


Z_dims = 2
X_dims = 784


class Generator():
    def __init__(self, **kwargs):
        self._theta = []
        self._W = {}
        self._b = {}
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.zeros_initializer
        with tf.variable_scope('generator'):
            self._W['1'] = tf.get_variable(
                'w1', [Z_dims, 128],
                initializer=w_init
            )
            self._b['1'] = tf.get_variable(
                'b1', [128], dtype=tf.float32,
                initializer=b_init
            )
            self._W['2'] = tf.get_variable(
                'w2', [128, X_dims], dtype=tf.float32,
                initializer=w_init
            )
            self._b['2'] = tf.get_variable(
                'b2', [X_dims], dtype=tf.float32,
                initializer=b_init
            )
        for k in self._W:
            self._theta.append(self._W[k])
            self._theta.append(self._b[k])
        # network
        self._inputs = tf.placeholder(
            tf.float32,
            [None, Z_dims]
        )
        self._hidden = tf.nn.relu(
            tf.nn.xw_plus_b(self._inputs, self._W['1'], self._b['1'])
        )
        self._probs = tf.sigmoid(
            tf.nn.xw_plus_b(self._hidden, self._W['2'], self._b['2'])
        )
        self._outputs = self._probs

    @property
    def inputs(self):
        return self._inputs

    @property
    def sample(self):
        return self._outputs

    @property
    def theta(self):
        return self._theta


class Discriminator():
    def __init__(self, **kwargs):
        self._theta = []
        self._W = {}
        self._b = {}
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.zeros_initializer
        with tf.variable_scope('discriminator'):
            self._W['1'] = tf.get_variable(
                'w1', [X_dims, 128], dtype=tf.float32,
                initializer=w_init
            )
            self._b['1'] = tf.get_variable(
                'b1', [128], dtype=tf.float32,
                initializer=b_init
            )
            self._W['2'] = tf.get_variable(
                'w2', [128, 1], dtype=tf.float32,
                initializer=w_init
            )
            self._b['2'] = tf.get_variable(
                'b2', [1], dtype=tf.float32,
                initializer=b_init
            )
        for k in self._W:
            self._theta.append(self._W[k])
            self._theta.append(self._b[k])

    def discriminate(self, inputs):
        hidden = tf.nn.relu(
            tf.nn.xw_plus_b(inputs, self._W['1'], self._b['1'])
        )
        logits = tf.nn.xw_plus_b(hidden, self._W['2'], self._b['2'])
        return logits

    @property
    def theta(self):
        return self._theta


class VanillaGAN():
    # G: the generator model
    # D: the discriminator model
    def __init__(self, **kwargs):
        # set the G and D model
        self._G = kwargs['G']
        self._D = kwargs['D']
        # the generated data from G
        self.Z = self._G.inputs
        self.X = tf.placeholder(
            tf.float32,
            self._G.sample.shape
        )
        self._fake_X = self._G.sample
        self._true_X = self.X
        self._fake_logits = self._D.discriminate(self._fake_X)
        self._true_logits = self._D.discriminate(self._true_X)
        # want the D to discriminate G and Data
        self._D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self._fake_logits),
                logits=self._fake_logits
            )
        )
        self._D_true_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._true_logits),
                logits=self._true_logits
            )
        )
        self.D_loss = tf.add(
            self._D_fake_loss,
            self._D_true_loss
        )
        # want to make the G to generate realistic data
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._fake_logits),
                logits=self._fake_logits
            )
        )
        self.D_optim = kwargs['optimizer'].minimize(
            self.D_loss, var_list=self._D.theta)
        self.G_optim = kwargs['optimizer'].minimize(
            self.G_loss, var_list=self._G.theta)

    @property
    def sample(self):
        return self._fake_X


if __name__ == '__main__':
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])

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

    mb_size = 128
    G = Generator()
    D = Discriminator()
    optim = tf.train.AdamOptimizer()
    GAN = VanillaGAN(G=G, D=D, optimizer=optim)

    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    if not os.path.exists('out/'):
        os.makedirs('out/')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    i = 0

    for it in range(1000000):
        if it % 1000 == 0:
            samples = sess.run(
                GAN.sample,
                feed_dict={GAN.Z: sample_Z(16, Z_dims)}
            )

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run(
            [GAN.D_optim, GAN.D_loss],
            feed_dict={GAN.X: X_mb, GAN.Z: sample_Z(mb_size, Z_dims)})
        _, G_loss_curr = sess.run(
            [GAN.G_optim, GAN.G_loss],
            feed_dict={GAN.Z: sample_Z(mb_size, Z_dims)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
