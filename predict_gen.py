import os, sys, glob, pathlib
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from absl import flags, app

import utils
import dataIO as io
from network import *
from model import conditional_alphaGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings

# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("indir", "\\\\tera.simizlab\\user\\lu\\taking_over\\data\\tfrecord\\C_LSDM\\CCR_100\\test", "directory included TFRecord")
flags.DEFINE_string("dir", "\\\\tera.simizlab\\user\\lu\\taking_over\\result\\C_LSDM\\c_alphaGAN", "model dir and outdir")
flags.DEFINE_string("gpu_index", "0", "GPU index")

flags.DEFINE_float("scale_lambda", 20., "scale parameter of L1 loss")
flags.DEFINE_float("scale_kappa", 20., "scale parameter of level set loss")
flags.DEFINE_float("scale_psi", 1., "scale parameter of eikonal loss")

flags.DEFINE_integer("k_size", 3, "kernel size of eikonal loss (should be odd)")
flags.DEFINE_integer("latent_dim", 9, "latent dim")
flags.DEFINE_integer("batch_size", 10, "batch size")
flags.DEFINE_integer("points_num", 1, "number of points")

flags.DEFINE_integer("model_index", 9300, "index of model")
flags.DEFINE_integer("num_of_test", 40, "number of test data")

flags.DEFINE_list("image_size", [48, 64, 80, 1], "image size")

def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # check folder
    if not os.path.exists(FLAGS.dir):
        raise Exception("model dirctory is not existed!")
    if not os.path.exists(os.path.join(FLAGS.dir, 'generalization')):
        os.makedirs(os.path.join(FLAGS.dir, 'generalization'))

    # get tfrecord list
    test_data_list = glob.glob(FLAGS.indir + '/*')

    # test step
    test_step = FLAGS.num_of_test // FLAGS.batch_size
    if FLAGS.num_of_test % FLAGS.batch_size != 0:
        test_step += 1

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list)
    test_set = test_set.map(
        lambda x: utils._parse_function_val_test(x, image_size=FLAGS.image_size),
        num_parallel_calls=os.cpu_count())
    test_set = test_set.batch(FLAGS.batch_size)
    test_iter = test_set.make_one_shot_iterator()
    test_data = test_iter.get_next()

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:
        # set network
        kwargs = {
            'sess': sess,
            'latent_dim': FLAGS.latent_dim,
            'scale_lambda': FLAGS.scale_lambda,
            'scale_kappa': FLAGS.scale_kappa,
            'scale_psi': FLAGS.scale_psi,
            'image_size': FLAGS.image_size,
            'points_num': FLAGS.points_num,
            'k_size': FLAGS.k_size,
            'encoder_layer': encoder_layer,
            'points_encoder_layer': points_encoder_layer,
            'generator_layer': generator_layer,
            'discriminator_layer': discriminator_layer,
            'code_discriminator_layer': code_discriminator_layer,
            'is_training': False
        }

        Model = conditional_alphaGAN(**kwargs)

        sess.run(init_op)

        # print parameters
        utils.cal_parameter()

        # test
        Model.restore_model(FLAGS.dir + '/model/model_{}'.format(FLAGS.model_index))
        tbar = tqdm(range(test_step), ascii=True)
        for i in tbar:
            test_image_batch, test_points_batch, test_label_batch = sess.run(test_data)
            reconstruction_batch = Model.reconstruction(test_image_batch, test_points_batch)

            # dilation of points
            test_points_batch = tf.keras.layers.MaxPooling3D(pool_size=5, strides=1, padding='same')(test_points_batch)
            test_points_batch = test_points_batch.eval()
            test_points_batch = test_points_batch * 2   # scaling

            if i is 0:
                test_label = np.asarray(test_label_batch)
                reconstruction = np.asarray(reconstruction_batch)[0]
                points = np.asarray(test_points_batch)
            else:
                test_label = np.concatenate((test_label, np.asarray(test_label_batch)), axis=0)
                reconstruction = np.concatenate((reconstruction, np.asarray(reconstruction_batch)[0]), axis=0)
                points = np.concatenate((points, np.array(test_points_batch)), axis=0)

        # calculate Jaccard Index and output images
        generalization = []
        tbar = tqdm(range(reconstruction.shape[0]), ascii=True)
        for i in tbar:
            test_label_single = test_label[i][:, :, :, 0]
            reconstruction_single = reconstruction[i][:, :, :, 0]
            points_single = points[i][:, :, :, 0]

            # label
            rec_label = np.where(reconstruction_single > 0.5, 0, 1)
            rec_label = rec_label.astype(np.int8)

            # calculate ji
            generalization.append([utils.jaccard(rec_label, test_label_single)])

            # label and points
            label_and_points = rec_label + points_single

            rec_label = rec_label.astype(np.int8)
            label_and_points = label_and_points.astype(np.int8)

            # output image
            io.write_mhd_and_raw(reconstruction_single, '{}.mhd'.format(
                os.path.join(FLAGS.dir, 'generalization', 'logodds', 'generate_{}'.format(i))),
                                 spacing=[1, 1, 1], origin=[0, 0, 0], compress=True)
            io.write_mhd_and_raw(rec_label,
                                 '{}.mhd'.format(
                                     os.path.join(FLAGS.dir, 'generalization', 'predict', 'recon_{}'.format(i))),
                                 spacing=[1, 1, 1], origin=[0, 0, 0], compress=True)
            io.write_mhd_and_raw(label_and_points, '{}.mhd'.format(
                os.path.join(FLAGS.dir, 'generalization', 'label_and_points', 'generate_{}'.format(i))),
                                 spacing=[1, 1, 1], origin=[0, 0, 0], compress=True)

        print('generalization = %f' % np.mean(generalization))

        # write csv
        io.write_csv(generalization,
                     os.path.join(FLAGS.dir, 'generalization', 'generalization_val_{}.csv'.format(FLAGS.model_index)),
                     'generalization')


if __name__ == '__main__':
    app.run(main)
