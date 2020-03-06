import os
import tensorflow as tf
import numpy as np
import cupy as cp
from scipy.stats import truncnorm

# load tfrecord function
def _parse_function(record, image_size = (256, 256, 1)):
    keys_to_features = {
        'level_set': tf.FixedLenFeature(np.prod(image_size), tf.float32),
        'points_image': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    level_set = parsed_features['level_set']
    points_image = parsed_features['points_image']

    level_set = tf.reshape(level_set, list(image_size))
    points_image = tf.reshape(points_image, image_size)
    return level_set, points_image

# load tfrecord function for val data and testing data
def _parse_function_val_test(record, image_size = (256, 256, 1)):
    keys_to_features = {
        'level_set': tf.FixedLenFeature(np.prod(image_size), tf.float32),
        'points_image': tf.FixedLenFeature(np.prod(image_size), tf.float32),
        'label':  tf.FixedLenFeature(np.prod(image_size), tf.int64)
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    level_set = parsed_features['level_set']
    points_image = parsed_features['points_image']
    label = parsed_features['label']

    level_set = tf.reshape(level_set, list(image_size))
    points_image = tf.reshape(points_image, image_size)
    label = tf.reshape(label, list(image_size))
    return level_set, points_image, label

# session config
def config(index = "0"):
    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            visible_device_list=index , # specify GPU number
            allow_growth=True
        )
    )
    return config

# calculate total parameters
def cal_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # cupy
    im1_cp = cp.asarray(im1)
    im2_cp = cp.asarray(im2)
    JI = np.double(cp.sum(cp.bitwise_and(im1_cp, im2_cp))) / np.double(cp.sum(cp.bitwise_or(im1_cp, im2_cp)))
    return JI

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def truncated_noise_sample(batch_size=1, dim_z=128, truncation=2., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return values
