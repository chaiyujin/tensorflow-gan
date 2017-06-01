import tensorflow as tf


class VanillaGAN():
    # Z_dims: the dim for the random noise
    # G: the generator model
    # D: the discriminator model
    def __init__(self, **kwargs):
        # set the input
        self.Z = tf.placeholder(
            tf.float32,
            [None, kwargs['Z_dims']]
        )
        # set the G and D model
        self._G = kwargs['G']
        self._D = kwargs['D']
        # the generated data from G
        self._fake_data = self._G.sample(self.Z)
        # want to make the G to generate realistic data
        self._G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self._G.logits),
                logits=self._G.logits
            )
        )
        # want the D to discriminate G and Data
        self._D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self._G.logits),
                logits=self._G.logits
            )
        )
        self._D_true_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like()
            )
        )
