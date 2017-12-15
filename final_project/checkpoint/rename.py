import sys, getopt

import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

reader = NewCheckpointReader('yolo_tiny.ckpt')
var_shapes = reader.get_variable_to_shape_map()
old_vars = {}

for var in var_shapes:
    old_vars[var] = tf.get_variable(var, var_shapes[var])

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'yolo_tiny.ckpt')
    new_vars = {}
    for var in old_vars:
        new_var = var.replace('biases', 'bias')
        new_var = new_var.replace('weights', 'kernel')
        new_var = new_var.replace('local', 'fc')
        new_vars[new_var] = old_vars[var]

    new_saver = tf.train.Saver(new_vars)
    new_saver.save(sess, 'yolo_tiny_renamed.ckpt')

reader = NewCheckpointReader('yolo_tiny_renamed.ckpt')
for var in reader.get_variable_to_shape_map():
    print var
