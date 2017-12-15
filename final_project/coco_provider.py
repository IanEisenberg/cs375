from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import collections
import os, sys
import time
import numpy as np
import tensorflow as tf

from tfutils import base, data, optimizer

import json
import copy

from tensorflow.python.ops import control_flow_ops

original_labels = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
       39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
       78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

max_objects = 96

labels_dict = {}
for idx, lab in enumerate(original_labels):
    labels_dict[lab] = idx + 1 # 0 is the background class

def _scale_image(image, image_height, image_width):
    ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [image_height, image_width], align_corners=True)
    image = tf.squeeze(image, axis=[0])

    return image, image_height / ih, image_width / iw

# Build data provider for COCO dataset
class COCO(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_height = 224,
                 crop_width = 224,
                 *args,
                 **kwargs):
        self.group = group
        self.batch_size = batch_size
        self.crop_height = crop_height
        self.crop_width = crop_width

        key_list = ['height', 'images', 'labels', 'num_objects', \
                    'segmentation_masks', 'width', 'bboxes']

        # key_list = ['images']
        source_dirs = ['/datasets/mscoco2/{}_tfrecords/{}/' .format(self.group, v) for v in key_list]
        
        BYTES_KEYs = {'images', 'labels', 'segmentation_masks', 'bboxes'}

        meta_dicts = [{v : {'dtype': tf.string, 'shape': []}} if v in BYTES_KEYs else {v : {'dtype': tf.int64, 'shape': []}} for v in key_list]

        super(COCO, self).__init__(
            source_dirs = source_dirs,
            meta_dicts = meta_dicts,
            batch_size=batch_size,
            n_threads=n_threads,
            *args, 
            **kwargs)

    def prep_data(self, data):
        for i in range(len(data)):
            d = data[i]
            batch_tensors = collections.defaultdict(list)
            for j in range(self.batch_size):
                inputs = {k: d[k][j] for k in d}
                image = inputs['images']
                image = tf.decode_raw(image, tf.uint8)
                ih = inputs['height']
                iw = inputs['width']
                ih = tf.cast(ih, tf.int32)
                iw = tf.cast(iw, tf.int32)
                inputs['height'] = ih
                inputs['width'] = iw
                bboxes = tf.decode_raw(inputs['bboxes'], tf.float64)
                imsize = tf.size(image)

                image = tf.cond(tf.equal(imsize, ih * iw), \
                      lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
                      lambda: tf.reshape(image, (ih, iw, 3)))
                
                image_height = ih
                image_width = iw
                num_instances = inputs['num_objects']
                num_instances = tf.cast(num_instances, tf.int32)
                inputs['num_objects'] = num_instances
                
                labels = tf.decode_raw(inputs['labels'], tf.int32)

                image, h_scale, w_scale = _scale_image(image, self.crop_height, self.crop_width)

                box_vector = tf.reshape(bboxes, [-1])
                zero_pad = tf.zeros([max_objects*4] - tf.shape(box_vector), dtype=box_vector.dtype)
                padded_boxes = tf.concat([box_vector, zero_pad], axis=0)
                padded_boxes = tf.reshape(padded_boxes, [-1, 4])

                # x1, y1, x2, y2 = tf.unstack(padded_boxes, axis=1)
                y1, x1, y2, x2 = tf.unstack(padded_boxes, axis=1)
                # y1, y2 = tf.cast(ih, tf.float64) - y1, tf.cast(ih, tf.float64)-y2
                # x1, x2 = tf.cast(iw, tf.float64) - x1, tf.cast(iw, tf.float64)-x2
                x1 = tf.minimum(tf.cast(self.crop_width - 1, tf.float64), x1 * w_scale)
                x2 = tf.minimum(tf.cast(self.crop_width - 1, tf.float64), x2 * w_scale)
                y1 = tf.minimum(tf.cast(self.crop_height - 1, tf.float64), y1 * h_scale)
                y2 = tf.minimum(tf.cast(self.crop_height - 1, tf.float64), y2 * h_scale)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                w = tf.abs(x2 - x1)
                h = tf.abs(y2 - y1)
                # x_center = tf.Print(x_center, [iw, ih, x1[0], x2[0], w_scale, x_center[0], w[0], padded_boxes[:num_instances]], message='bboxes', summarize=8)
                padded_boxes = tf.stack([x_center, y_center, w, h], axis=1)

                zero_pad = tf.zeros([max_objects] - tf.shape(labels), dtype=labels.dtype)
                padded_labels = tf.reshape(tf.cast(tf.concat([labels, zero_pad], axis=0), tf.float64), [-1, 1])

                # padded_labels = tf.cast(tf.reshape(tf.pad(labels, [[0, max_objects]]), [-1, 1]), tf.float64)
                # ones = tf.ones([tf.shape(padded_boxes)[0], 1], dtype=padded_boxes.dtype)
                padded_boxes_with_conf = tf.concat([padded_boxes, padded_labels], 1) #tf.pad(padded_boxes, tf.constant([[0,0],[0,1]]), constant_values=1.0)
                padded_boxes_with_conf.set_shape([max_objects, 5])
                image.set_shape([self.crop_height, self.crop_width, 3])
                # padded_boxes_with_conf = tf.Print(padded_boxes_with_conf, [num_instances, ih, iw])
                example_values = {'coco_images': image, 'boxes': padded_boxes_with_conf, 'num_objects': num_instances, 'ih': ih, 'iw': iw}#, 'multiple_labels': labels}
                for k, v in example_values.iteritems():
                    batch_tensors[k].append(v)

            data[i] = {k: tf.stack(v) for k, v in batch_tensors.iteritems()}

        return data


    def init_ops(self):
        self.input_ops = super(COCO, self).init_ops()
        self.input_ops = self.prep_data(self.input_ops)

        return self.input_ops
