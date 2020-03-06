import os, sys, glob, pathlib
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from absl import flags, app
from scipy.stats import truncnorm

import utils
import dataIO as io
from network import *
from model import conditional_alphaGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings

# flag
FLAGS = flags.FLAGS
flags.DEFINE_string("indir", "\\\\tera.simizlab\\user\\lu\\taking_over\\data\\tfrecord\\C_LSDM\\CCR_100\\pa_test", "directory included TFRecord")
flags.DEFINE_string("dir", "\\\\tera.simizlab\\user\\lu\\taking_over\\result\\C_LSDM\\c_alphaGAN", "model dir and outdir")
flags.DEFINE_string("ground_truth", "\\\\tera.simizlab\\user\\lu\\taking_over\\data\\test_image\\probabilistic_altas\\test_label.txt", "directory included ground truth (.mhd)")
flags.DEFINE_string("gpu_index", "0", "GPU index")

flags.DEFINE_float("scale_lambda", 20., "scale parameter of L1 loss")
flags.DEFINE_float("scale_kappa", 20., "scale parameter of level set loss")
flags.DEFINE_float("scale_psi", 1., "scale parameter of eikonal loss")

flags.DEFINE_integer("k_size", 3, "kernel size of eikonal loss (should be odd)")
flags.DEFINE_integer("latent_dim", 9, "latent dim")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_integer("points_num", 1, "number of points")

flags.DEFINE_integer("model_index", 9300, "index of model")
flags.DEFINE_integer("num_of_test", 40, "number of test data")
flags.DEFINE_integer("num_of_generate", 500, "number of generate data")

flags.DEFINE_list("image_size", [48, 64, 80, 1], "image size")


def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # check folder
    if not os.path.exists(FLAGS.dir):
        raise Exception("model dirctory is not existed!")
    if not os.path.exists(os.path.join(FLAGS.dir, 'dice')):
        os.makedirs(os.path.join(FLAGS.dir, 'dice'))

    # get ground truth list
    ground_truth_list = io.load_list(FLAGS.ground_truth)

    # load ground truth
    ground_truth = io.load_data_from_path(ground_truth_list, dtype='int32')

    # get tfrecord list
    test_data_list = glob.glob(FLAGS.indir + '/*')

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list)
    test_set = test_set.map(
        lambda x: utils._parse_function_val_test(x, image_size=FLAGS.image_size),
        num_parallel_calls=os.cpu_count())
    test_set = test_set.repeat()
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
        dice_list = []

        Model.restore_model(FLAGS.dir + '/model/model_{}'.format(FLAGS.model_index))
        for i in range(FLAGS.num_of_test):
            _, test_points_batch, _ = sess.run(test_data)

            np.random.seed(4)

            tbar = tqdm(range(FLAGS.num_of_generate//FLAGS.batch_size), ascii=True)
            for j in tbar:

                z = np.random.normal(0., 1., size=[FLAGS.batch_size, FLAGS.latent_dim])
                # z = utils.truncated_noise_sample(FLAGS.batch_size, FLAGS.latent_dim, truncation=2.0)
                generate_batch = Model.generate_sample(z, test_points_batch)

                # save logodds
                generate_batch_ = np.asarray(generate_batch)
                generate_batch_ = generate_batch_[0, :, :, :]
                for image_index in range(generate_batch_.shape[0]):
                    gen = generate_batch_[image_index][:, :, :, 0]
                    io.write_mhd_and_raw(gen, '{}.mhd'.format(
                        os.path.join(FLAGS.dir, 'dice', '{}'.format(i),
                                     '{}'.format(j * FLAGS.batch_size + image_index))),
                                         spacing=[1, 1, 1],
                                         origin=[0, 0, 0], compress=True)

                if j is 0:
                    data = np.asarray(generate_batch)[0]
                    label = np.where(data > 0.5, 0, 1)
                    label = label.astype(np.int8)
                    pa = np.sum(label, axis=0)
                else:
                    data = np.asarray(generate_batch)[0]
                    label_ = np.where(data > 0.5, 0, 1)
                    label_ = label_.astype(np.int8)
                    pa = pa + np.sum(label_, axis=0)

            pa = pa / float(FLAGS.num_of_generate)
            pa = pa.astype(np.float32)

            # output image
            io.write_mhd_and_raw(pa, '{}_{}.mhd'.format(
                os.path.join(FLAGS.dir, 'dice', 'PA'), i), spacing=[1, 1, 1], origin=[0, 0, 0], compress=True)

            # dice
            gt = ground_truth[i]
            gt = gt.astype(np.float32)
            dice = utils.dice_coef(gt, pa)
            dice_list.append([round(dice, 6)])
            print(dice)

        print('dice = %f' % np.mean(dice_list))
        # write csv
        io.write_csv(dice_list,
                     os.path.join(FLAGS.dir, 'dice', 'dice_{}.csv'.format(FLAGS.model_index)),
                     'dice')


if __name__ == '__main__':
    app.run(main)
