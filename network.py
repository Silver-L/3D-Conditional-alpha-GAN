'''
# Network Architecture of 3D conditional alpha-GAN
# Author: Zhihui Lu
# Date: 2020/01/22
# tf.keras.layers will cause some problems !
# Using tf.layers instead of tf.keras.layers
'''

import tensorflow as tf
import numpy as np


def generator_layer(input, points_encoded, shape=(3, 4, 5, 512)):
    dense_shape = points_encoded[-1].get_shape().as_list()
    x = dense_op(input, units=np.prod(shape) - dense_shape[-1], activation=tf.nn.relu)
    x = tf.concat((x, points_encoded[-1]), axis=-1)
    x = tf.reshape(x, shape=[-1] + list(shape))

    shape_1 = points_encoded[-2].get_shape().as_list()
    x_1 = deconv_sn(x, channels=256-shape_1[-1], kernel=5, stride=2, padding='SAME', scope='gen_0_0')
    x_2 = deconv_sn(x, channels=shape_1[-1], kernel=5, stride=2, padding='SAME', scope='gen_0_1')
    x_p = tf.multiply(x_2, points_encoded[-2])
    x = tf.concat((x_1, x_p), axis=-1)

    x = deconv_sn(x, channels=128, kernel=5, stride=2, padding='SAME', scope='gen_1')

    shape_2 = points_encoded[-3].get_shape().as_list()
    x_1 = deconv_sn(x, channels=64-shape_2[-1], kernel=5, stride=2, padding='SAME', scope='gen_2_0')
    x_2 = deconv_sn(x, channels=shape_2[-1], kernel=5, stride=2, padding='SAME', scope='gen_2_1')
    x_p = tf.multiply(x_2, points_encoded[-3])
    x = tf.concat((x_1, x_p), axis=-1)

    x = deconv_sn(x, channels=32, kernel=5, stride=1, padding='SAME', scope='gen_3')

    x = tf.layers.conv3d_transpose(x, filters=1, kernel_size=5, strides=2, padding='same',
                                   activation=tf.nn.sigmoid)

    return x


def discriminator_layer(input, drop_rate=0.7):
    x = conv_sn(input, channels=64, kernel=5, stride=2, pad=1, scope='dis_0')
    x = conv_sn(x, channels=128, kernel=5, stride=2, pad=1, scope='dis_1')
    x = conv_sn(x, channels=256, kernel=5, stride=2, pad=1, scope='dis_2')
    x = conv_sn(x, channels=512, kernel=5, stride=2, pad=1, scope='dis_3')
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = dense_op(x, units=1000, activation=tf.nn.leaky_relu, drop_rate=drop_rate)
    x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
    return x


def code_discriminator_layer(input, drop_rate=0.7):
    x = dense_op(input, units=500, activation=tf.nn.leaky_relu, drop_rate=drop_rate)
    x = dense_op(x, units=500, activation=tf.nn.leaky_relu, drop_rate=drop_rate)
    x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
    return x


def encoder_layer(input, points_encoded, latent_dim, is_training=True, drop_rate=0.7):
    # block 1
    block_1 = conv_op(input, kernel_size=5, filters=64, strides=2, is_training=is_training, drop_rate=drop_rate)
    GAP_b1 = tf.keras.layers.GlobalAveragePooling3D()(block_1)

    # block 2
    shape_1 = points_encoded[0].get_shape().as_list()
    block_2_1 = conv_op(block_1, kernel_size=5, filters=128 - shape_1[-1], strides=2, is_training=is_training,
                        drop_rate=drop_rate)
    block_2_2 = conv_op(block_1, kernel_size=5, filters=shape_1[-1], strides=2, is_training=is_training,
                        drop_rate=drop_rate)
    block_2_2 = tf.multiply(block_2_2, points_encoded[0])
    block_2 = tf.concat((block_2_1, block_2_2), axis=-1)
    GAP_b2 = tf.keras.layers.GlobalAveragePooling3D()(block_2)

    # block 3
    block_3 = conv_op(block_2, kernel_size=5, filters=256, strides=2, is_training=is_training, drop_rate=drop_rate)
    GAP_b3 = tf.keras.layers.GlobalAveragePooling3D()(block_3)

    # block 4
    shape_2 = points_encoded[1].get_shape().as_list()
    block_4_1 = conv_op(block_3, kernel_size=5, filters=256-shape_2[-1], strides=2, is_training=is_training, drop_rate=drop_rate)
    block_4_2 = conv_op(block_3, kernel_size=5, filters=shape_2[-1], strides=2, is_training=is_training, drop_rate=drop_rate)
    block_4_2 = tf.multiply(block_4_2, points_encoded[1])
    block_4 = tf.concat((block_4_1, block_4_2), axis=-1)

    block_4 = conv_op(block_4, kernel_size=5, filters=512, strides=1, is_training=is_training, drop_rate=drop_rate)
    GAP = tf.keras.layers.GlobalAveragePooling3D()(block_4)

    # concat
    GAP = tf.concat((GAP, GAP_b1), axis=-1)
    GAP = tf.concat((GAP, GAP_b2), axis=-1)
    GAP = tf.concat((GAP, GAP_b3), axis=-1)

    output = dense_op(GAP, units=1000, activation=tf.nn.leaky_relu, drop_rate=drop_rate)
    output = tf.layers.dense(output, units=latent_dim)
    return output

def points_encoder_layer(input, is_training=True, drop_rate=0.7):
    x = input
    filter_list = [20, 40, 80, 80]
    encoder_concat_list = [1, 3]
    generator_concat_list = [0, 2]
    encoder_concat = []
    generator_concat = []

    for i in range(len(filter_list)):
        x = conv_op(x, kernel_size=5, filters=filter_list[i], strides=2, is_training=is_training, drop_rate=drop_rate)
        if i in encoder_concat_list:
            encoder_concat.append(x)
        if i in generator_concat_list:
            generator_concat.append(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    generator_concat.append(x)

    return encoder_concat, generator_concat


# convolution operation
def conv_op(input, filters, kernel_size=5, strides=2, padding='same', is_training=True, drop_rate=0.7):
    output = tf.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                              activation=tf.nn.leaky_relu)(input)
    output = tf.layers.dropout(output, drop_rate)
    output = batch_norm(output, is_training=is_training)
    return output


# deconvolution operation
def deconv_op(input, filters, kernel_size=3, strides=2, padding='same', is_training=True, drop_rate=0.7):
    output = tf.layers.conv3d_transpose(input, filters=filters, kernel_size=kernel_size,
                                        strides=strides, padding=padding, activation=tf.nn.relu)
    output = tf.layers.dropout(output, drop_rate)
    output = batch_norm(output, is_training=is_training)
    return output


def conv_sn(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left
            pad_front = pad // 2
            pad_back = pad - pad_front

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_front, pad_back], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_front, pad_back], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                           mode='REFLECT')

        w = tf.get_variable("kernel", shape=[kernel, kernel, kernel, x.get_shape()[-1], channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv3d(input=x, filter=spectral_norm(w),
                         strides=[1, stride, stride, stride, 1], padding='VALID')
        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        x = tf.nn.leaky_relu(x)

        return x


def deconv_sn(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [tf.shape(x)[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0),
                            x_shape[3] * stride + max(kernel - stride, 0), channels]

        w = tf.get_variable("kernel", shape=[kernel, kernel, kernel, channels, x.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv3d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                   strides=[1, stride, stride, stride, 1], padding=padding)

        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        x = tf.nn.relu(x)

        return x


def dense_op(input, units, activation=tf.nn.leaky_relu, drop_rate=0.7):
    output = tf.layers.dense(input, units=units, activation=activation)
    output = tf.layers.dropout(output, drop_rate)
    return output


# batch normalization
def batch_norm(x, is_training=True):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
