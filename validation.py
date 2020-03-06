import os, glob, random, pathlib, sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from absl import flags, app

import utils
import dataIO as io
from network import *
from model import conditional_alphaGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Surpress verbose warnings


FLAGS = flags.FLAGS
flags.DEFINE_string("indir", "\\\\tera.simizlab\\user\\lu\\taking_over\\data\\tfrecord\\C_LSDM\\CCR_100\\val", "directory included TFRecord")
flags.DEFINE_string("dir", "\\\\tera.simizlab\\user\\lu\\taking_over\\result\\C_LSDM\\c_alphaGAN\\debug", "model dir and outdir")
flags.DEFINE_string("ground_truth", '\\\\tera.simizlab\\user\\lu\\taking_over\\data\\val_image\\full_components\\down\\64\\label', "directory included ground truth (.mhd)")
flags.DEFINE_string("gpu_index", "0", "GPU index")

flags.DEFINE_float("scale_lambda", 20., "scale parameter of L1 loss")
flags.DEFINE_float("scale_kappa", 30., "scale parameter of level set loss")
flags.DEFINE_float("scale_psi", 1., "scale parameter of eikonal loss")

flags.DEFINE_integer("k_size", 3, "kernel size of eikonal loss (should be odd)")
flags.DEFINE_integer("latent_dim", 9, "latent dim")
flags.DEFINE_integer("points_num", 1, "number of points")
flags.DEFINE_integer("batch_size", 13, "batch size")

flags.DEFINE_integer("train_iteration", 12001, "number of training iteration")
flags.DEFINE_integer("num_of_val", 39, "number of validation data")
flags.DEFINE_integer("save_model_step", 150, "step of saving model")
flags.DEFINE_integer("num_of_generate", 300, "number of generate data")

flags.DEFINE_list("image_size", [48, 64, 80, 1], "image size")

def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # check folder
    if not os.path.exists(os.path.join(FLAGS.dir, 'tensorboard')):
        os.makedirs(os.path.join(FLAGS.dir, 'tensorboard'))
    if not os.path.exists(FLAGS.dir):
        raise Exception("model dirctory is not existed!")

    # get tfrecord list
    val_data_list = glob.glob(FLAGS.indir + '/*')

    # get ground truth list
    ground_truth_list = glob.glob(FLAGS.ground_truth + '/*.mhd')

    # load ground truth
    ground_truth = io.load_data_from_path(ground_truth_list, dtype='int32')

    # number of model
    num_of_model = FLAGS.train_iteration // FLAGS.save_model_step
    num_of_model = num_of_model + 1 if FLAGS.train_iteration % FLAGS.save_model_step is not 0 else num_of_model - 1

    # val_iter
    num_val_iter = FLAGS.num_of_val // FLAGS.batch_size
    if FLAGS.num_of_val % FLAGS.batch_size != 0:
        num_val_iter += 1

    # load val data
    val_set = tf.data.Dataset.list_files(val_data_list)
    val_set = val_set.apply(
        tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=os.cpu_count()))
    val_set = val_set.map(lambda x: utils._parse_function_val_test(x, image_size=FLAGS.image_size),
                              num_parallel_calls=os.cpu_count())
    val_set = val_set.repeat()
    val_set = val_set.batch(FLAGS.batch_size)
    val_set = val_set.make_one_shot_iterator()
    val_data = val_set.get_next()

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

        # print parameters
        utils.cal_parameter()

        # prepare tensorboard
        writer_gen = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'val_generalization'))
        writer_spe = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'val_specificity'))
        writer_val_ls = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'val_ls'))
        writer_val_eikonal = tf.summary.FileWriter(os.path.join(FLAGS.dir, 'tensorboard', 'val_eikonal'))
        writer_all = tf.summary.FileWriter(
            os.path.join(FLAGS.dir, 'tensorboard', 'val_all'))  # mean of generalization and specificity

        # saving loss operation
        value_loss = tf.Variable(0.0)
        tf.summary.scalar("evaluation", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        # validation
        tbar = tqdm(range(num_of_model), ascii=True)
        for step in tbar:
            Model.restore_model(FLAGS.dir + '/model/model_{}'.format(step*FLAGS.save_model_step))

            generalization, specificity, val_ls, val_eikonal = [], [], [], []
            points = []
            real_image = []

            # reconstruction
            for i in range(num_val_iter):
                val_image_batch, val_points_batch, val_label_batch = sess.run(val_data)
                points.append(val_points_batch)
                real_image.append(val_image_batch)

                reconstruction_batch = Model.reconstruction(val_image_batch, val_points_batch)

                if i is 0:
                    val_label = np.asarray(val_label_batch)
                    reconstruction = np.asarray(reconstruction_batch)[0]
                else:
                    val_label = np.concatenate((val_label, np.asarray(val_label_batch)), axis=0)
                    reconstruction = np.concatenate((reconstruction, np.asarray(reconstruction_batch)[0]), axis=0)

            # calculate generalization
            for i in range(reconstruction.shape[0]):
                val_label_single = val_label[i][:, :, :, 0]
                reconstruction_single = reconstruction[i][:, :, :, 0]

                # label
                rec_label = np.where(reconstruction_single > 0.5, 0, 1)

                # calculate ji
                generalization.append([utils.jaccard(rec_label, val_label_single)])

            # samples from latent space
            points_ls = np.ones_like(points) * 0.5
            for i in range(FLAGS.num_of_generate // FLAGS.batch_size):
                shuffle_fun(points_ls, points)

                z = np.random.normal(0., 1., size=[FLAGS.batch_size, FLAGS.latent_dim])
                generate_batch, level_set_loss, eikonal_loss = Model.validation_specificity(
                    points_ls[random.randint(0, num_val_iter - 1)], z, points[random.randint(0, num_val_iter - 1)])

                val_ls.append(level_set_loss)
                val_eikonal.append(eikonal_loss)

                if i is 0:
                    samples = np.asarray(generate_batch)
                else:
                    samples = np.concatenate((samples, np.asarray(generate_batch)), axis=0)

            # calculate specificity
            for i in range(samples.shape[0]):
                gen = samples[i][:, :, :, 0]

                # label
                gen_label = np.where(gen > 0.5, 0, 1)

                # calculate ji
                case_max_ji = 0.
                for image_index in range(ground_truth.shape[0]):
                    ji = utils.jaccard(gen_label, ground_truth[image_index])
                    if ji > case_max_ji:
                        case_max_ji = ji
                specificity.append([case_max_ji])

            s = "val_generalization: {:.4f}, val_specificity: {:.4f}, ls: {:.4f}, eikonal: {:.4f}, mean: {:.4f}".format(
                np.mean(generalization),
                np.mean(specificity),
                np.mean(val_ls),
                np.mean(val_eikonal),
                (np.mean(generalization) + np.mean(specificity)) / 2.)

            tbar.set_description(s)

            summary_gen = sess.run(merge_op, {value_loss: np.mean(generalization)})
            summary_spe = sess.run(merge_op, {value_loss: np.mean(specificity)})
            summary_ls = sess.run(merge_op, {value_loss: np.mean(val_ls)})
            summary_eikonal = sess.run(merge_op, {value_loss: np.mean(val_eikonal)})
            summary_all = sess.run(merge_op, {value_loss: (np.mean(generalization) + np.mean(specificity)) / 2.})

            writer_gen.add_summary(summary_gen, step * FLAGS.save_model_step)
            writer_spe.add_summary(summary_spe, step * FLAGS.save_model_step)
            writer_val_ls.add_summary(summary_ls, step * FLAGS.save_model_step)
            writer_val_eikonal.add_summary(summary_eikonal, step * FLAGS.save_model_step)
            writer_all.add_summary(summary_all, step * FLAGS.save_model_step)

            generalization.clear()
            specificity.clear()
            val_ls.clear()
            val_eikonal.clear()
            points.clear()
            real_image.clear()


# shuffle list in the same order
def shuffle_fun(a, b):
    data_list = list(zip(a, b))
    random.shuffle(data_list)
    a, b = zip(*data_list)
    return np.asarray(a), np.asarray(b)

if __name__ == '__main__':
    app.run(main)