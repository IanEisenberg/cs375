from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from net import Net 

class YoloTinyNet(Net):

  def __init__(self, common_params, net_params, test=False):
    """
    common params: a params dict
    net_params   : a params dict
    """
    super(YoloTinyNet, self).__init__(common_params, net_params)
    #process params
    self.image_size = int(common_params['image_size'])
    self.num_classes = int(common_params['num_classes'])
    self.cell_size = int(net_params['cell_size'])
    self.boxes_per_cell = int(net_params['boxes_per_cell'])
    self.batch_size = int(common_params['batch_size'])
    self.weight_decay = float(net_params['weight_decay'])

    if not test:
      self.object_scale = float(net_params['object_scale'])
      self.noobject_scale = float(net_params['noobject_scale'])
      self.class_scale = float(net_params['class_scale'])
      self.coord_scale = float(net_params['coord_scale'])

  def yolo_shared(self, images, reuse=False):
    """
    Args:
        images: NxWxHxC tensor
        reuse: passed to tf.layers calls, whether or not to reuse weights
    Returns:
        outputs: dict of layer activations
    """
    # Shared portion of the ImageNet / COCO pipeline
    outputs = {} 
    conv_num = 1
    temp_conv = tf.layers.conv2d(images, 16, (3, 3), (1, 1),  
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1

    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool1')

    temp_conv = tf.layers.conv2d(temp_pool, 32, (3, 3), (1, 1),  
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1

    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool2')

    temp_conv = tf.layers.conv2d(temp_pool, 64, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1
    
    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool3')

    temp_conv = tf.layers.conv2d(temp_pool, 128, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1

    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool4')

    temp_conv = tf.layers.conv2d(temp_pool, 256, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1

    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool5')

    temp_conv = tf.layers.conv2d(temp_pool, 512, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1

    temp_pool = tf.layers.max_pooling2d(temp_conv, [2, 2], [2, 2], padding='SAME', name='pool6')

    temp_conv = tf.layers.conv2d(temp_pool, 1024, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1     

    temp_conv = tf.layers.conv2d(temp_conv, 1024, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1 

    temp_conv = tf.layers.conv2d(temp_conv, 1024, (3, 3), (1, 1), 
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        padding='SAME', name='conv' + str(conv_num),
        reuse=reuse)
    outputs['conv_%s' % conv_num] = temp_conv
    conv_num += 1 

    temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2))

    # Fully connected layers
    conv_flat = tf.contrib.layers.flatten(temp_conv)
    local1 = tf.layers.dense(conv_flat, 256,
        activation=tf.nn.relu, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        name='fc1',
        reuse=reuse)
    outputs['fc1'] = local1

    local2 = tf.layers.dense(local1, 4096,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        name='fc2',
        reuse=reuse)
    outputs['fc2'] = local2

    return outputs

  def inference(self, inputs, train=True, norm=True, **kwargs):
    """Build the yolo model

    Args:
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    Returns:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    
    outputs = inputs
    # Get the activations for each input image, then combine the dicts
    imagenet_input = inputs['images']
    coco_input = inputs['coco_images']
    for v in (imagenet_input, coco_input):
        v.set_shape([self.batch_size, self.image_size, self.image_size, 3])

    imagenet_output = self.yolo_shared(imagenet_input)
    coco_output = self.yolo_shared(coco_input, reuse=True)

    for key in imagenet_output.keys():
      outputs['imagenet_%s' % key] = imagenet_output[key]
      outputs['coco_%s' % key] = coco_output[key]

    # Bounding box outputs for COCO image
    local3 = tf.layers.dense(outputs['coco_fc2'], self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5), name='fc3')
    n1 = self.cell_size * self.cell_size * self.num_classes

    n2 = n1 + self.cell_size * self.cell_size * self.boxes_per_cell

    class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes))
    scales = tf.reshape(local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell))
    boxes = tf.reshape(local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))

    local3 = tf.concat([class_probs, scales, boxes], 3)

    predicts = local3
    outputs['bboxes'] = predicts

    # Logit outputs for Imagenet image
    logits = tf.layers.dense(outputs['imagenet_fc2'], 1000, 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), 
        name='logits')
    outputs['logits'] = logits

    return outputs, {}

  def validation(self, inputs, train=False, norm=True, **kwargs):
    outputs = inputs
    images = inputs['images']
    layer_output = self.yolo_shared(images)
    for key in layer_output.keys():
      outputs[key] = layer_output[key]
    return outputs, {}

  def iou(self, boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    #calculate the left up point
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection
    intersection = rd - lu 

    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def cond1(self, num, object_num, loss, predict, label, nilboy):
    """
    if num < object_num
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels, nilboy):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, width, height, class)
    """
    label = labels[num:num+1, :] 
    label = tf.cast(tf.reshape(label, [-1]), tf.float32)

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    min_x = (label[0] - label[2] / 2) / (self.image_size // self.cell_size)
    max_x = (label[0] + label[2] / 2) / (self.image_size // self.cell_size)
    min_y = (label[1] - label[3] / 2) / (self.image_size // self.cell_size)
    max_y = (label[1] + label[3] / 2) / (self.image_size // self.cell_size)

    min_x = tf.maximum(tf.floor(min_x), 0)
    min_y = tf.maximum(tf.floor(min_y), 0)

    max_x = tf.ceil(max_x) #tf.maximum(tf.ceil(max_x), min_x+1)
    max_y = tf.ceil(max_y) #tf.maximum(tf.ceil(max_y), min_y+1)

    temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    objects = tf.ones(temp, tf.float32)

    temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    # objects is a [cell_size, cell_size] tensor with 1's where the ground truth object is
    objects = tf.pad(objects, temp, "CONSTANT") 

    #calculate responsible tensor [CELL_SIZE, CELL_SIZE]
    center_x = label[0] / (self.image_size / self.cell_size)
    center_x = tf.floor(center_x)

    center_y = label[1] / (self.image_size / self.cell_size)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size -center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    # response is a [cell_size, cell_size] tensor with a single 1 in the cell at the center of the bounding box
    response = tf.pad(response, temp, "CONSTANT")     #objects = response

    #calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    ## Predictions are initially scaled by [ cell_px, cell_px, w, h ] (i.e. in [0, 1] ranges). Rescale them to pixel coordinates.
    predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]
    predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])
    predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

    ## After scaling, apply translation to fully convert to image pixel coordinate space.
    base_boxes = np.zeros([self.cell_size, self.cell_size, 4])
    for y in range(self.cell_size):
      for x in range(self.cell_size):
        #nilboy
        base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
    base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])
    predict_boxes = base_boxes + predict_boxes

    iou_predict_truth = self.iou(predict_boxes, label[0:4])
    #calculate C [cell_size, cell_size, boxes_per_cell]
    C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))
    
    max_I = tf.reduce_max(I, 2, keep_dims=True)

    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I 


    p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

    #calculate truth x,y,sqrt_w,sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))
    #sqrt_w = tf.abs(label[2])
    #sqrt_h = tf.abs(label[3])

    #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    #p_sqrt_w = tf.sqrt(tf.abs(predict_boxes[:, :, :, 2])) * ((tf.cast(predict_boxes[:, :, :, 2] > 0, tf.float32) * 2) - 1)
    #p_sqrt_h = tf.sqrt(tf.abs(predict_boxes[:, :, :, 3])) * ((tf.cast(predict_boxes[:, :, :, 3] > 0, tf.float32) * 2) - 1)
    #p_sqrt_w = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 2]))
    #p_sqrt_h = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 3]))
    #p_sqrt_w = predict_boxes[:, :, :, 2]
    #p_sqrt_h = predict_boxes[:, :, :, 3]
    p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
    #calculate truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:self.num_classes]

    #class_loss
    class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
    #object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

    #noobject_loss
    #noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    #coord_loss
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size))) * self.coord_scale # +
                 #tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                 #tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    # nilboy = I

    return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels, nilboy



  def loss(self, predicts, labels, objects_num):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)
    loss = [0, 0, 0, 0]
    objects_num = tf.reshape(objects_num, [-1])
    for i in range(self.batch_size):
      predict = predicts[i, :, :, :]
      label = labels[i, :, :]
      object_num = objects_num[i]
      p = tf.Print(objects_num, [object_num, label, tf.shape(predict)], message = 'Number of instances', summarize=60)
      nilboy = tf.ones([7,7,2])
      args = [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss], predict, label, nilboy]
      tuple_results = tf.while_loop(self.cond1, self.body1, args)
      for j in range(4):
        loss[j] = loss[j] + tuple_results[2][j]
      nilboy = tuple_results[5]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size)

    tf.summary.scalar('class_loss', loss[0]/self.batch_size)
    tf.summary.scalar('object_loss', loss[1]/self.batch_size)
    tf.summary.scalar('noobject_loss', loss[2]/self.batch_size)
    tf.summary.scalar('coord_loss', loss[3]/self.batch_size)
    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size )

    return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy, p
