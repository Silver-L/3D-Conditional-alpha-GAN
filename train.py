import os, sys, pathlib, random, glob
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
flags.DEFINE_string("indir", "\\\\tera.simizlab\\user\\lu\\taking_over\\data\\tfrecord\\C_LSDM\\CCR_100\\train", "directory included TFRecord")
flags.DEFINE_string("outdir", "\\\\tera.simizlab\\user\\lu\\taking_over\\result\\C_LSDM\\c_alphaGAN\\debug", "output directory")
flags.DEFINE_string("gpu_index", "0", "GPU index")

flags.DEFINE_float("scale_lambda", 300., "scale parameter of L1 loss")
flags.DEFINE_float("scale_kappa", 1., "scale parameter of level set loss")
flags.DEFINE_float("scale_psi", 1., "scale parameter of eikonal loss")
flags.DEFINE_float("lr", 3e-5, "learning rate")

flags.DEFINE_integer("k_size", 3, "kernel size of eikonal loss (should be odd)")
flags.DEFINE_integer("latent_dim", 9, "latent dim")
flags.DEFINE_integer("points_num", 1, "number of points")
flags.DEFINE_integer("batch_size", 20, "batch size")

flags.DEFINE_integer("num_iteration", 12001, "number of iteration")
flags.DEFINE_integer("e_g_step", 1,
                     "step of training encoder and generator (>=1), make sure that d_loss does not go to zero ")
flags.DEFINE_integer("d_step", 1, "step of training discriminator")
flags.DEFINE_integer("save_loss_step", 150, "step of saving loss")
flags.DEFINE_integer("save_model_step", 150, "step of saving model")
flags.DEFINE_integer("shuffle_buffer_size", 200, "buffer size of shuffle")

flags.DEFINE_list("image_size", [48, 64, 80, 1], "image size")

def main(argv):

    # turn off log message
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # check folder
    if not os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard')):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard'))
    if not os.path.exists(os.path.join(FLAGS.outdir, 'model')):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))

    # save flag file
    FLAGS.flags_into_string()
    FLAGS.append_flags_into_file(os.path.join(FLAGS.outdir, 'flagfile.txt'))

    # get tfrecord list
    train_data_list = glob.glob(FLAGS.indir + '/*')
    # shuffle list
    random.shuffle(train_data_list)

    # load train data
    train_set = tf.data.Dataset.list_files(train_data_list)
    train_set = train_set.apply(
        tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=os.cpu_count()))
    train_set = train_set.map(
        lambda x: utils._parse_function(x, image_size=FLAGS.image_size),
        num_parallel_calls=os.cpu_count())
    train_set = train_set.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_set = train_set.repeat()
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

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
            'k_size': FLAGS.k_size,
            'image_size': FLAGS.image_size,
            'points_num': FLAGS.points_num,
            'encoder_layer': encoder_layer,
            'points_encoder_layer': points_encoder_layer,
            'generator_layer': generator_layer,
            'discriminator_layer': discriminator_layer,
            'code_discriminator_layer': code_discriminator_layer,
            'lr': FLAGS.lr,
            'is_training': True
        }

        Model = conditional_alphaGAN(**kwargs)

        # print parameters
        utils.cal_parameter()

        # prepare tensorboard
        writer_e_loss = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'e_loss'), sess.graph)
        writer_g_loss = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'g_loss'))
        writer_d_loss = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'd_loss'))
        writer_c_loss = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'c_loss'))

        # saving loss operation
        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        step_e, step_g, step_d, step_c, = [], [], [], []

        # training
        tbar = tqdm(range(FLAGS.num_iteration), ascii=True)
        for step in tbar:
            for i in range(FLAGS.e_g_step):
                train_image_batch, points_batch = sess.run(train_data)
                noise = np.random.normal(0., 1., size=[FLAGS.batch_size, FLAGS.latent_dim])
                e_loss = Model.update_e(train_image_batch, points_batch)
                g_loss = Model.update_g(train_image_batch, points_batch, noise)

            for i in range(FLAGS.d_step):
                d_loss = Model.update_d(train_image_batch, points_batch, noise)

            c_loss = Model.update_c(train_image_batch, points_batch, noise)

            step_e.append(e_loss)
            step_g.append(g_loss)
            step_d.append(d_loss)
            step_c.append(c_loss)

            if step % FLAGS.save_loss_step is 0:
                s = "e_loss: {:.4f}, g_loss: {:.4f}, d_loss: {:.4f}, c_loss: {:.4f}".format(
                    np.mean(step_e),
                    np.mean(step_g),
                    np.mean(step_d),
                    np.mean(step_c))
                tbar.set_description(s)

                summary_e = sess.run(merge_op, {value_loss: np.mean(step_e)})
                summary_g = sess.run(merge_op, {value_loss: np.mean(step_g)})
                summary_d = sess.run(merge_op, {value_loss: np.mean(step_d)})
                summary_c = sess.run(merge_op, {value_loss: np.mean(step_c)})

                writer_e_loss.add_summary(summary_e, step)
                writer_g_loss.add_summary(summary_g, step)
                writer_d_loss.add_summary(summary_d, step)
                writer_c_loss.add_summary(summary_c, step)

                step_e.clear()
                step_g.clear()
                step_d.clear()
                step_c.clear()

            if step % FLAGS.save_model_step is 0:
                # save model
                Model.save_model(FLAGS.outdir, step)

if __name__ == '__main__':
    app.run(main)