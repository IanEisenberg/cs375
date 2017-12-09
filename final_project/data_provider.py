from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

import coco_provider
from ImageNetDataProvider import ImageNetDataProvider

sys.path.append('../no_tfutils/')

class Combine_world:

    def __init__(self,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 cfg_dataset={},
                 nfromd=0,
                 depthnormal=0,
                 categorymulti=1,
                 queue_params=None,
                 withflip=0, 
                 with_noise=0, 
                 noise_level=10,
                 no_shuffle=0,
                 crop_each=0,
                 onlyflip_cate=0,
                 flipnormal=0,
                 eliprep=0,
                 thprep=0,
                 crop_time=5,
                 crop_rate=5,
                 replace_folder=None,
                 as_list=0,
                 sub_mean=0,
                 mean_path=None,
                 with_color_noise=0,
                 *args, **kwargs
                 ):
        self.group = group
        self.batch_size = batch_size
        self.categorymulti = categorymulti
        self.ret_list = categorymulti>1 or as_list==1
        self.queue_params = queue_params
        self.withflip = withflip
        self.crop_each = crop_each

        self.shuffle_flag = group=='train'

        if no_shuffle==1:
            self.shuffle_flag = False

        if self.ret_list:
            assert not self.queue_params==None, "Must send queue params in"

            self.queue_params_list = []
            self.data_params_list = []

        self.crop_size = 224
        if not crop_size==None:
            self.crop_size = crop_size

        self.all_providers = []
        if cfg_dataset.get('imagenet', 0)==1:            
            # set data path
            imagenet_data_path = '/datasets/TFRecord_Imagenet_standard'
            
            curr_batch_size = batch_size
            trans_dicts_imagenet = [{'images': 'images_labelnet'}, {'labels': 'labels_imagenet'}]
            if self.ret_list:
                curr_data_params = {
                    'func': data.TFRecordsParallelByFileProvider,
                    'source_dirs': source_dirs_imagenet,
                    'trans_dicts': trans_dicts_imagenet,
                    'postprocess': postprocess_imagenet,
                    'batch_size': batch_size,
                    'n_threads': n_threads,
                    'shuffle': self.shuffle_flag,
                    'file_pattern': file_pattern,
                }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)

                curr_queue_params = copy.deepcopy(self.queue_params)
                curr_queue_params['batch_size'] = curr_queue_params['batch_size']*self.categorymulti
                self.queue_params_list.append(curr_queue_params)
            else:
                self.all_providers.append(ImageNetDataProvider(data_path = imagenet_data_path,
                                                               group = group,
                                                               batch_size = batch_size,
                                                               n_thread = n_threads,
                                                               crop_size = self.crop_size,
                                                               crop_height = self.crop_size,
                                                               crop_width = self.crop_size,
                                                               *args, **kwargs
                                                ))

        if cfg_dataset.get('coco', 0)==1:
            key_list = ['height', 'images', 'labels', 'num_objects', \
                    'segmentation_masks', 'width']
            trans_dicts_coco = [{'images': 'images_coco'}, 
                                    {'ih': 'ih'},
                                    {'boxes': 'boxes'},
                                    {'num_objects': 'num_objects'},
                                    {'iw': 'iw'}]
            if self.ret_list:
                curr_data_params = {
                        'func': coco_provider.COCO,
                        'source_dirs': source_dirs,
                        'meta_dicts': meta_dicts,
                        'group': group,
                        'batch_size': batch_size,
                        'n_threads': n_threads,
                        'image_min_size': 240,
                        'crop_height': 224,
                        'crop_width': 224,
                        }
                curr_data_params.update(kwargs)
                self.data_params_list.append(curr_data_params)
                self.queue_params_list.append(self.queue_params)
            else:
                self.all_providers.append(coco_provider.COCO(group = group,
                                                             batch_size = batch_size,
                                                             n_threads = n_threads,
                                                             image_min_size = 240,
                                                             crop_height = self.crop_size,
                                                             crop_width = self.crop_size,
                                                             *args, **kwargs
                                                            ))


    def postproc_label(self, labels):

        curr_batch_size = self.batch_size

        labels.set_shape([curr_batch_size])
        
        if curr_batch_size==1:
            labels = tf.squeeze(labels, axis = [0])

        return labels


    def postprocess_images(self, ims, dtype_now, shape_now):
        def _postprocess_images(im):
            im = tf.image.decode_png(im, dtype = dtype_now)
            im.set_shape(shape_now)
            if dtype_now==tf.uint16:
                im = tf.cast(im, tf.int32)
            return im
        if dtype_now==tf.uint16:
            write_dtype = tf.int32
        else:
            write_dtype = dtype_now
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=write_dtype)

    def postprocess_normalize(self, ims):
        def _postprocess_normalize(im):
            im = tf.cast(im, tf.float32)
            mean, var = tf.nn.moments(im, axes = list(range(len(im.get_shape().as_list()) - 1)))
            #var = tf.Print(var, [var], message = 'Var')
            #print(var.get_shape().as_list())
            #print(mean.get_shape().as_list())
            mean = tf.Print(mean, [mean], message = 'Mean')
            im = im - mean
            im = im / (var + 0.001)
            mean, var = tf.nn.moments(im, axes = list(range(len(im.get_shape().as_list()))))
            mean = tf.Print(mean, [mean], message = 'Mean after')
            var = tf.Print(var, [var], message = 'Var after')
            im = im - mean
            im = im / (var + 0.001)
            return im

        def _postprocess_normalize_2(im):
            im = tf.cast(im, tf.float32)
            #print(im.get_shape().as_list())
            im = tf.image.per_image_standardization(im)
            return im

        #return tf.map_fn(lambda im: _postprocess_normalize(im), ims, dtype = tf.float32)
        #return tf.map_fn(lambda im: _postprocess_normalize_2(im), ims, dtype = tf.float32)
        if self.batch_size==1:
            return _postprocess_normalize_2(ims)
        else:
            return tf.map_fn(lambda im: _postprocess_normalize_2(im), ims, dtype = tf.float32)
    
    def postprocess_resize(self, ims, newsize_1=240, newsize_2=320):
        return tf.image.resize_images(ims, (newsize_1, newsize_2))

    def postprocess_rawdepth(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, tf.int32)
            im = tf.reshape(im, [240, 320, 1])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.int32)

    def init_ops(self):
        all_init_ops = [data_temp.init_ops() for data_temp in self.all_providers]
        num_threads = len(all_init_ops[0])

        self.ret_init_ops = []
        for indx_t in xrange(num_threads):
            curr_dict = {}
            for curr_init_ops in all_init_ops:
                curr_dict.update(curr_init_ops[indx_t])
            self.ret_init_ops.append(curr_dict)
        return self.ret_init_ops
