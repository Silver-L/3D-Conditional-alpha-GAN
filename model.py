'''
# 3D conditional-alpha-GAN Model
# Author: Zhihui Lu
# Date: 2020/01/22
'''

import os
import tensorflow as tf
import numpy as np


class conditional_alphaGAN(object):

    def __init__(self, sess, latent_dim, scale_lambda, scale_kappa, scale_psi, image_size, points_num, k_size,
                 encoder_layer, points_encoder_layer, generator_layer, discriminator_layer, code_discriminator_layer,
                 is_training=True, lr=1e-4):
        self._sess = sess
        self._latent_dim = latent_dim
        self._image_size = image_size
        self._points_num = points_num
        self._k_size = k_size
        self._lambda = scale_lambda
        self._kappa = scale_kappa
        self._psi = scale_psi
        self._is_training = is_training
        self._encoder_layer = encoder_layer
        self._points_encoder_layer = points_encoder_layer
        self._generator_layer = generator_layer
        self._discriminator_layer = discriminator_layer
        self._code_discriminator_layer = code_discriminator_layer
        self._lr = lr

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('input'):
            with tf.variable_scope('noise'):
                self._noise_input = tf.placeholder(tf.float32, shape=[None, self._latent_dim])

            with tf.variable_scope('image'):
                self._real_image_input = tf.placeholder(tf.float32, shape=[None] + list(self._image_size))

            with tf.variable_scope('points'):
                self._points_image = tf.placeholder(tf.float32, shape=[None] + list(self._image_size))

        # encode points image
        self._points_encoder, self._points_generator = self.points_encoder(points_image=self._points_image,
                                                                           is_training=self._is_training)

        # encode image
        self._z_hat = self.encoder(image=self._real_image_input, points_info=self._points_encoder,
                                   latent_dim=self._latent_dim, is_training=self._is_training)

        # reconstruction
        self._rec = self.generator(latent_var=self._z_hat, points_info=self._points_generator)

        # random sample
        self._gen = self.generator(latent_var=self._noise_input, points_info=self._points_generator)

        # discriminator
        self._d_real = self.discriminator(image=self._real_image_input)
        self._d_rec = self.discriminator(image=self._rec)
        self._d_gen = self.discriminator(image=self._gen)

        # code discriminator
        self._c_z = self.code_discriminator(latent_var=self._noise_input)
        self._c_z_hat = self.code_discriminator(latent_var=self._z_hat)

        # loss
        self._e_loss = self.encoder_loss(real=self._real_image_input, reconstruction=self._rec,
                                         points=self._points_image, c_z_hat=self._c_z_hat, scale_lambda=self._lambda,
                                         scale_kappa=self._kappa, scale_psi=self._psi, image_size=self._image_size,
                                         points_num=self._points_num, k_size=self._k_size)

        self._g_loss = self.generator_loss(real=self._real_image_input,
                                           reconstruction=self._rec, sample=self._gen,
                                           points=self._points_image,
                                           d_rec=self._d_rec,
                                           d_gen=self._d_gen,
                                           scale_lambda=self._lambda,
                                           scale_kappa=self._kappa,
                                           scale_psi=self._psi,
                                           image_size=self._image_size,
                                           points_num=self._points_num,
                                           k_size=self._k_size)

        self._d_loss = self.discriminator_loss(d_real=self._d_real, d_rec=self._d_rec, d_gen=self._d_gen)

        self._c_loss = self.code_discriminator_loss(c_z=self._c_z, c_z_hat=self._c_z_hat)

        # validation
        self._val_ls = self.level_set_loss(real=self._real_image_input, fake=self._gen,
                                           points=self._points_image, image_size=self._image_size,
                                           points_num=self._points_num)
        self._val_eikonal = self.eikonal_loss_points(fake=self._gen, points=self._points_image,
                                                     k_size=self._k_size)

        # parameters
        self.e_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        self.p_e_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Points_Encoder')
        self.g_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        self.d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        self.c_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Code_Discriminator')

        # optimizer
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.5, beta2=0.9)

        # update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # update encoder (eta) and points encoder (zeta)
            self._train_e = optimizer.minimize(self._e_loss, var_list=self.e_var + self.p_e_var)

            # update generator (theta)
            self._train_g = optimizer.minimize(self._g_loss, var_list=self.g_var + self.p_e_var)

            # update discriminator (phi)
            self._train_d = optimizer.minimize(self._d_loss, var_list=self.d_var)

            # update code discriminator (omega)
            self._train_c = optimizer.minimize(self._c_loss, var_list=self.c_var)

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)

    def encoder(self, image, points_info, latent_dim, is_training=True, drop_rate=0.7, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.variable_scope('Encoder', reuse=reuse):
            output = self._encoder_layer(input=image, points_encoded=points_info, latent_dim=latent_dim,
                                         is_training=is_training, drop_rate=drop_rate)
            return output

    def points_encoder(self, points_image, is_training=True, drop_rate=0.7, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.variable_scope('Points_Encoder', reuse=reuse):
            output = self._points_encoder_layer(input=points_image, is_training=is_training, drop_rate=drop_rate)
            return output

    def generator(self, latent_var, points_info, shape=(3, 4, 5, 512), reuse=tf.compat.v1.AUTO_REUSE):
        with tf.variable_scope('Generator', reuse=reuse):
            output = self._generator_layer(input=latent_var, points_encoded=points_info, shape=shape)
            return output

    def discriminator(self, image, drop_rate=0.7, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.variable_scope('Discriminator', reuse=reuse):
            output = self._discriminator_layer(input=image, drop_rate=drop_rate)
            return output

    def code_discriminator(self, latent_var, drop_rate=0.7, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.variable_scope('Code_Discriminator', reuse=reuse):
            output = self._code_discriminator_layer(input=latent_var, drop_rate=drop_rate)
            return output

    def encoder_loss(self, real, reconstruction, points, c_z_hat, scale_lambda, scale_kappa, scale_psi, image_size,
                     points_num, k_size):
        with tf.variable_scope('encoder_loss'):
            # L1 loss
            L1 = scale_lambda * self.abs_criterion(real, reconstruction) + \
                 scale_kappa * self.level_set_loss(real, reconstruction, points, image_size, points_num) + \
                 scale_psi * self.eikonal_loss_points(reconstruction, points, k_size=k_size)

            # code loss
            c_loss = self.density_ratio(c_z_hat)

            # encoder loss
            e_loss = tf.reduce_mean(L1 + c_loss)
            return e_loss

    def generator_loss(self, real, reconstruction, sample, points, d_rec, d_gen, scale_lambda, scale_kappa, scale_psi,
                       image_size, points_num, k_size):
        with tf.variable_scope('generator_loss'):
            # reconstruction loss
            rec_loss = tf.reduce_mean(scale_lambda * self.abs_criterion(real, reconstruction) +
                                      scale_kappa * self.level_set_loss(real, reconstruction, points, image_size,
                                                                        points_num) +
                                      scale_psi * self.eikonal_loss_points(reconstruction, points,
                                                                           k_size=k_size) + self.density_ratio(d_rec))

            # generation loss
            gen_loss = tf.reduce_mean(self.density_ratio(d_gen)
                                      + scale_kappa * self.level_set_loss(real, sample, points, image_size, points_num)
                                      + scale_psi * self.eikonal_loss_points(sample, points, k_size=k_size))
            # generator loss
            g_loss = rec_loss + gen_loss

            return g_loss

    def discriminator_loss(self, d_real, d_rec, d_gen, eps=1e-6):
        with tf.variable_scope('discriminator_loss'):
            # treat p*(x) as real
            real_loss = -tf.log(d_real + eps)

            # treat reconstructions as fake
            rec_loss = -tf.log(1 - d_rec + eps)

            # treat generated samples as fake
            gen_loss = -tf.log(1 - d_gen + eps)

            # discriminator loss
            d_loss = tf.reduce_mean(real_loss + rec_loss) + tf.reduce_mean(gen_loss)
            return d_loss

    def code_discriminator_loss(self, c_z, c_z_hat, eps=1e-6):
        with tf.variable_scope('code_discriminator_loss'):
            # treat codes from variational distribution as fake
            z_hat_loss = -tf.log(1 - c_z_hat + eps)

            # treat p(z) as real
            z_loss = -tf.log(c_z + eps)

            # code discriminator loss
            c_loss = tf.reduce_mean(z_hat_loss) + tf.reduce_mean(z_loss)
            return c_loss

    def level_set_loss(self, real, fake, points, image_size, points_num = 1):

        real_points = tf.multiply(real, points)
        rec_points = tf.multiply(fake, points)
        ls_loss = self.abs_criterion(real_points, rec_points) * np.prod(image_size) / float(points_num)

        return ls_loss

    def eikonal_loss_contour(self, fake, range_value=(-10., 10.), alpha=0.1):
        '''
        contour version
        calculate eikonal loss near the contour of the image
        '''

        # inverse sigmoid transform
        image = self.inverse_sigmoid_function(fake, alpha=alpha)

        # get the mask of the contour
        image_shape = tf.shape(image)
        ones_matrix = tf.ones(image_shape)
        zeros_matrix = tf.zeros(image_shape)
        condition = image

        mask = tf.where(condition > range_value[0], ones_matrix, zeros_matrix)
        mask = tf.where(condition < range_value[1], mask, zeros_matrix)

        # calculate eikonal loss
        nabla = tf.numpy_function(self.gradient_3d, [image], [tf.float32, tf.float32, tf.float32])
        nabla_x, nabla_y, nabla_z = tf.math.square(nabla[0]), tf.math.square(nabla[1]), tf.math.square(nabla[2])

        # calculate eikonal loss inner the range
        nabla_sum = nabla_x + nabla_y + nabla_z
        nabla_sum = tf.math.sqrt(nabla_sum)
        loss = tf.math.abs(nabla_sum - 1.)

        loss = tf.multiply(loss, mask)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3, 4])

        # compute number of nonzero elements in mask images
        num = tf.math.count_nonzero(mask, axis=[1, 2, 3, 4])
        num = tf.cast(num, tf.float32)
        loss = loss / num

        return loss

    def eikonal_loss_points(self, fake, points, k_size=3, alpha=0.1):
        '''
        calculate eikonal loss near the points
        '''

        # inverse sigmoid transform
        image = self.inverse_sigmoid_function(fake, alpha=alpha)

        # get the mask (neighbour of points)
        mask = self.dilation(points, k_size)

        # calculate eikonal loss
        nabla = tf.numpy_function(self.gradient_3d, [image], [tf.float32, tf.float32, tf.float32])
        nabla_x, nabla_y, nabla_z = tf.math.square(nabla[0]), tf.math.square(nabla[1]), tf.math.square(nabla[2])

        # calculate eikonal loss inner the range
        nabla_sum = nabla_x + nabla_y + nabla_z
        nabla_sum = tf.math.sqrt(nabla_sum)
        loss = tf.math.abs(nabla_sum - 1.)

        loss = tf.multiply(loss, mask)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3, 4])

        # compute number of nonzero elements in mask images
        num = tf.math.count_nonzero(mask, axis=[1, 2, 3, 4])
        num = tf.cast(num, tf.float32)
        loss = loss / num

        return loss

    def eikonal_loss(self, fake, alpha=0.1):
        '''
        whole image version
        calculate eikonal loss of the whole image
        '''

        # inverse sigmoid transform
        image = self.inverse_sigmoid_function(fake, alpha=alpha)

        nabla = tf.numpy_function(self.gradient_3d, [image], [tf.float32, tf.float32, tf.float32])
        nabla_x, nabla_y, nabla_z = tf.math.square(nabla[0]), tf.math.square(nabla[1]), tf.math.square(nabla[2])
        nabla_sum = tf.math.sqrt(nabla_x + nabla_y + nabla_z)
        loss = self.abs_criterion(nabla_sum, 1.)

        return loss

    @staticmethod
    def abs_criterion(real, reconstruction):
        with tf.variable_scope('L1_loss'):
            l1_loss = tf.reduce_mean(tf.abs(real - reconstruction), axis=[1, 2, 3, 4])
            return l1_loss

    @staticmethod
    def density_ratio(x, eps=1e-6):
        with tf.variable_scope('density_ratio'):
            ratio = -tf.log(x + eps) + tf.log(1 - x + eps)
            return ratio

    @staticmethod
    def inverse_sigmoid_function(y, alpha=0.1, eps=1e-6):
        x = tf.math.log(y / (1. - y + eps) + eps)
        x = (1. / alpha) * x
        return x

    @staticmethod
    def gradient_3d(x):
        return np.gradient(x, axis=[1, 2, 3])

    @staticmethod
    def dilation(x, k_size=3):
        x_dilated = tf.keras.layers.MaxPooling3D(pool_size=k_size, strides=1, padding='same')(x)
        return x_dilated

    def update_e(self, real_image, points):
        _, e_loss = self._sess.run([self._train_e, self._e_loss],
                                   feed_dict={self._real_image_input: real_image, self._points_image: points})
        return e_loss

    def update_g(self, real_image, points, noise):
        _, g_loss= self._sess.run([self._train_g, self._g_loss],
                                                        feed_dict={self._real_image_input: real_image,
                                                                   self._points_image: points,
                                                                   self._noise_input: noise})
        return g_loss

    def update_d(self, real_image, points, noise):
        _, d_loss = self._sess.run([self._train_d, self._d_loss],
                                   feed_dict={self._real_image_input: real_image, self._points_image: points,
                                              self._noise_input: noise})
        return d_loss

    def update_c(self, real_image, points, noise):
        _, c_loss = self._sess.run([self._train_c, self._c_loss],
                                   feed_dict={self._real_image_input: real_image, self._points_image: points,
                                              self._noise_input: noise})
        return c_loss

    def reconstruction(self, real_image, points):
        rec = self._sess.run([self._rec], feed_dict={self._real_image_input: real_image, self._points_image: points})
        return rec

    def generate_sample(self, noise, points):
        samples = self._sess.run([self._gen], feed_dict={self._noise_input: noise, self._points_image: points})
        return samples

    def generate_sample_debug(self, noise, points):
        samples, eikonal_loss = self._sess.run([self._gen, self._val_eikonal],
                                               feed_dict={self._noise_input: noise, self._points_image: points})
        return samples, eikonal_loss


    def save_model(self, path, index):
        save = self.saver.save(self._sess, os.path.join(path, 'model', 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)

    def validation_specificity(self, real, noise, points):
        sample, ls_loss, eikonal_loss = self._sess.run([self._gen, self._val_ls, self._val_eikonal],
                                                       feed_dict={self._real_image_input: real,
                                                                  self._points_image: points,
                                                                  self._noise_input: noise})
        return sample, ls_loss, eikonal_loss
