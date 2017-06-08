# Cycle GAN with Wasserstein Divergence
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from models import ConvNet, DeconvNet


default_config = {
    'b_size': 32,  # batch_size
    'z_dims': 100,  # z dims
    'img_h': 256,
    'img_w': 256,
    'img_c': 1,
    'penalty_lambda': 10,
    'norm_lambda': 5
}


class CycleGAN():
    def __init__(self, scope='CycleGAN', config=default_config):
        self._b_size = config['b_size']
        self._z_dims = config['z_dims']
        self._img_h = config['img_h']
        self._img_w = config['img_w']
        self._img_c = config['img_c']
        self._penalty_lambda = config['penalty_lambda']
        self._norm_lambda = config['norm_lambda']
        # input
        img_shape = [self._b_size, self._img_h, self._img_w, self._img_c]
        self._x = tf.placeholder(tf.float32, img_shape)
        self._y = tf.placeholder(tf.float32, img_shape)
        self._z = tf.placeholder(tf.float32, [self._b_size, self._z_dims])

        # X to Y
        Gxy = self.__G(self._x, self._z, 'Gxy')
        Dy_true = self.__D(self._y, 'Dy')
        Dy_fake = self.__D(Gxy, 'Dy', True)
        # penalty y
        eps_y = tf.random_uniform([], 0.0, 1.0, name='eps_y')
        penalty_y = eps_y * self._y + (1 - eps_y) * Gxy
        Dy_penalty = self.__D(penalty_y, 'Dy', True)
        ddy = tf.gradients(Dy_penalty, penalty_y)[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), axis=1))
        ddy = tf.reduce_mean(tf.square(ddy - 1.0) * self._penalty_lambda)
        WD_Dy = tf.reduce_mean(Dy_true) - tf.reduce_mean(Dy_fake) - ddy
        self._Dy_loss = -WD_Dy
        self._Gxy_loss = -tf.reduce_mean(Dy_fake)

        # Y to X
        Gyx = self.__G(self._y, self._z, 'Gyx')
        Dx_true = self.__D(self._x, 'Dx')
        Dx_fake = self.__D(Gyx, 'Dx', True)
        # penalty x
        eps_x = tf.random_uniform([], 0.0, 1.0, name='eps_x')
        penalty_x = eps_x * self._x + (1 - eps_x) * Gyx
        Dx_penalty = self.__D(penalty_x, 'Dx', True)
        ddx = tf.gradients(Dx_penalty, penalty_x)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self._penalty_lambda)
        WD_Dx = tf.reduce_mean(Dx_true) - tf.reduce_mean(Dx_fake) - ddx
        self._Dx_loss = -WD_Dx
        self._Gyx_loss = -tf.reduce_mean(Dx_fake)

        # Cycle
        Gxyx = self.__G(
            self.__G(self._x, self._z, 'Gxy', True),
            self._z, 'Gyx', True
        )
        Gyxy = self.__G(
            self.__G(self._y, self._z, 'Gyx', True),
            self._z, 'Gxy', True
        )
        L1_x = tf.reduce_mean(
            tf.losses.absolute_difference(
                labels=self._x,
                predictions=Gxyx,
                scope='L1_x'
            )
        ) * self._norm_lambda
        L1_y = tf.reduce_mean(
            tf.losses.absolute_difference(
                labels=self._y,
                predictions=Gyxy,
                scope='L1_y'
            )
        ) * self._norm_lambda
        self._Gxy_loss += L1_y
        self._Gyx_loss += L1_x

        self._Gxy = Gxy
        self._Gyx = Gyx

        self._saver = tf.train.Saver()

    @property
    def Gxy(self):
        return self._Gxy

    @property
    def Gyx(self):
        return self._Gyx

    def __G(self, x, z, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # Encoder
            net = ConvNet()
            # input 256x256, 3
            net.conv_layer(64, activation='lrelu')
            # => 128x128, 64
            net.conv_layer(128, activation='lrelu', batch_norm=True)
            # => 64x64, 128
            net.conv_layer(256, activation='lrelu', batch_norm=True)
            # => 32x32, 256
            net.conv_layer(512, activation='lrelu', batch_norm=True)
            # => 16x16, 512

            # Bottleneck
            conved = net(x)
            conved_shape = conved.get_shape()
            lin = tflayers.flatten(conved)
            lin_size = int(lin.get_shape()[1])
            lin = tflayers.fully_connected(
                lin, 1024, activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            code = tf.concat([lin, z], axis=1)
            lin = tflayers.fully_connected(
                code, lin_size, activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            bottleneck = tf.reshape(lin, conved_shape)

            # Decoder
            dec = DeconvNet()
            # input 16x16, 512
            dec.deconv_layer(256, activation='lrelu', batch_norm=True)
            # => 32x32, 256
            dec.deconv_layer(128, activation='lrelu', batch_norm=True)
            # => 64x64, 128
            dec.deconv_layer(64, activation='lrelu', batch_norm=True)
            # => 128x128, 64
            dec.deconv_layer(x.get_shape()[-1])

            logits = dec(bottleneck)
            image = tf.sigmoid(logits)

            return image

    def __D(self, x, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = ConvNet()
            # input 256x256, 3
            net.conv_layer(64, activation='lrelu')
            # => 128x128, 64
            net.conv_layer(128, activation='lrelu', batch_norm=True)
            # => 64x64, 128
            net.conv_layer(256, activation='lrelu', batch_norm=True)
            # => 32x32, 256
            net.conv_layer(512, activation='lrelu', batch_norm=True)
            # => 16x16, 512
            lin = tflayers.flatten(net(x))
            d = tflayers.fully_connected(
                lin, 1, activation_fn=None,
                weights_initializer=tf.random_normal_initializer(0, 0.02)
            )
            return d

    def train_minibatch(self, sess, x_batch, y_batch):
        pass


if __name__ == '__main__':
    cycle = CycleGAN()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./log',
                                             tf.get_default_graph())
        summary = sess.run(merged)
        train_writer.add_summary(summary, 0)
