# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

factor1 = cfg.SCORE_FACTOR1
factor2 = cfg.SCORE_FACTOR2

#threshold
rpn3_reject = cfg.RPN_REJECT3
rpn2_reject = cfg.RPN_REJECT2
rpn1_reject = cfg.RPN_REJECT1

reject3 = cfg.REJECT3
reject2 = cfg.REJECT2
reject1 = cfg.REJECT1

#factor
rpn3_reject_f = cfg.RPN_REJECT3_FACTOR
rpn2_reject_f = cfg.RPN_REJECT2_FACTOR
rpn1_reject_f = cfg.RPN_REJECT1_FACTOR

reject3_f = cfg.REJECT3_FACTOR
reject2_f = cfg.REJECT2_FACTOR
reject1_f = cfg.REJECT1_FACTOR

#batch
rpn_batch3 = cfg.TRAIN.RPN_BATCH3
rpn_batch2 = cfg.TRAIN.RPN_BATCH2
rpn_batch1 = cfg.TRAIN.RPN_BATCH1
rpn_batch = cfg.TRAIN.RPN_BATCH

batch3 = cfg.TRAIN.BATCH3
batch2 = cfg.TRAIN.BATCH2
batch1 = cfg.TRAIN.BATCH1
batch = cfg.TRAIN.BATCH

#OHEM
OHEM3 = cfg.TRAIN.OHEM3
OHEM2 = cfg.TRAIN.OHEM2
OHEM1 = cfg.TRAIN.OHEM1
OHEM = cfg.TRAIN.OHEM


class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

    self.endpoint = {}

  def build_network(self, sess, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                        trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
     
      #store conv3_3
      self.endpoint['conv3_3'] = net

      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')

      #store conv4_3
      self.endpoint['conv4_3'] = net

      #continue conv5
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      #store conv5_3
      self.endpoint['conv5_3'] = net
      self._layers['head'] = self.endpoint['conv5_3']

      # build the anchors for the image
      self._anchor_component()


      
      
      ##-----------------------------------------------rpn-----------------------------------------------------------------##
      rpn = slim.conv2d(self.endpoint['conv5_3'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")

      self._act_summaries.append(rpn)

      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score_pre')

      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')


      # rpn_cls_score = rpn3_cls_score*rpn3_cls_score_scale*0.25 + rpn2_cls_score*rpn2_cls_score_scale*0.25 + rpn1_cls_score*rpn1_cls_score_scale*0.25 + rpn0_cls_score*rpn0_cls_score_scale*0.25

      #used added up score
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")


      if is_training:
        #compute anchor loss       
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor", [], [], rpn_batch, OHEM)

      ######################################################RPN DONE##################################################################

      #---------------------------------------------------porposal is made here------------------------------------------------------#

      if is_training:
        # #compute anchor loss       
        # rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor", rpn1_reject_inds)
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], [])

        # with tf.control_dependencies([rpn_labels]):
        #   rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")

      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], [])
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], [])
        else:
          raise NotImplementedError

      #----------------------------------------------------------finish proposal-----------------------------------------------------#



      #############################################################RCNN START###############################################################


      #------------------------------------------------------rcnn 3----------------------------------------------------#
      # rcnn
      # generate target
      #if is_training:          
      #  with tf.control_dependencies([rpn_labels]):
      #    rois, _, passinds3 = self._proposal_target_layer(rois, roi_scores, "rpn3_rois", batch3)

      if cfg.POOLING_MODE == 'crop':
        pool31 = self._crop_pool_layer(self.endpoint['conv3_3'], rois, 16, 14, "pool31")
      else:
        raise NotImplementedError


      pool31_conv = slim.conv2d(pool31, 256, [1, 1], trainable=is_training, weights_initializer=initializer, scope="pool31_conv")
      pool31_avg = slim.avg_pool2d(pool31_conv, [14, 14], padding='SAME', scope='pool31_avg', stride = 1) 
      pool31_flat = slim.flatten(pool31_avg, scope='flatten31') 

      fc3_2 = slim.fully_connected(pool31_flat, 512, scope='fc3_2', weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training)

      # if is_training:
      #   fc3_2 = slim.dropout(fc3_2, keep_prob=0.5, is_training=True, scope='fc3_2')

      #combine
      scale3_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale3_2')
      fc_combine3_2 = tf.scalar_mul(scale3_2, fc3_2)

      cls3_score = slim.fully_connected(fc_combine3_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls3_score')
      #store RCNN3
      self._predictions["cls3_score"] = cls3_score

      cls3_prob = self._softmax_layer(cls3_score, "cls3_prob")

      #reject via threshold
      cls3_inds_1 = tf.reshape(tf.where(tf.less(cls3_prob[:,0], reject3)), [-1])
      rois = tf.gather(rois, tf.reshape(cls3_inds_1,[-1]))
      fc_combine3_2 = tf.gather(fc_combine3_2, tf.reshape(cls3_inds_1,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls3_inds_1,[-1]))

      #reject via factor
      _, cls3_inds_2 = tf.nn.top_k(cls3_score[:,0]*-1, tf.cast(tf.cast(tf.shape(cls3_score)[0], tf.float32)*tf.cast((1-reject3_f), tf.float32), tf.int32))
      rois = tf.gather(rois, tf.reshape(cls3_inds_2,[-1]))
      fc_combine3_2 = tf.gather(fc_combine3_2, tf.reshape(cls3_inds_2,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls3_inds_2,[-1]))

      #self._act_summaries.append(self.endpoint['conv4_2'])
      
      #------------------------------------------------------rcnn 2----------------------------------------------------#
      #generate target
      if is_training:          
        with tf.control_dependencies([rpn_labels]):
          roi_scores = tf.gather(roi_scores, tf.reshape(cls3_inds_1,[-1]))
          roi_scores = tf.gather(roi_scores, tf.reshape(cls3_inds_2,[-1]))
      #    rois, _, passinds4 = self._proposal_target_layer(rois, roi_scores, "rpn2_rois", batch2)
      #    cls3_score = tf.gather(cls3_score, tf.reshape(passinds4,[-1]))
      #    fc_combine3_2 = tf.gather(fc_combine3_2, tf.reshape(passinds4,[-1]))

      if cfg.POOLING_MODE == 'crop':
        pool41 = self._crop_pool_layer(self.endpoint['conv4_3'], rois, 8, 14, "pool41")
      else:
        raise NotImplementedError


      pool41_conv = slim.conv2d(pool41, 256, [1, 1], trainable=is_training, weights_initializer=initializer, scope="pool41_conv")
      pool41_avg = slim.avg_pool2d(pool41_conv, [14, 14], padding='SAME', scope='pool41_avg', stride = 1) 
      pool41_flat = slim.flatten(pool41_avg, scope='flatten41') 

      fc4_2 = slim.fully_connected(pool41_flat, 512, scope='fc4_2', weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training)

      # if is_training:
      #   fc4_2 = slim.dropout(fc4_2, keep_prob=0.5, is_training=True, scope='fc4_2')

      fc4_2 = self._score_add_up(fc_combine3_2, fc4_2, factor1, factor2, 'fc_42_comb')

      #combine
      scale4_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale4_2')

      fc_combine4_2 = tf.scalar_mul(scale4_2, fc4_2)

      cls4_score = slim.fully_connected(fc_combine4_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls4_score')


      #store RCNN2
      self._predictions["cls2_score"] = cls4_score

      cls4_prob = self._softmax_layer(cls4_score, "cls4_prob")


      #reject via threshold
      cls4_inds_1 = tf.reshape(tf.where(tf.less(cls4_prob[:,0], reject2)), [-1])
      rois = tf.gather(rois, tf.reshape(cls4_inds_1,[-1]))
      fc_combine4_2 = tf.gather(fc_combine4_2, tf.reshape(cls4_inds_1,[-1]))
      cls4_score = tf.gather(cls4_score, tf.reshape(cls4_inds_1,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls4_inds_1,[-1]))

      #reject via factor
      _, cls4_inds_2 = tf.nn.top_k(cls4_score[:,0], tf.cast(tf.cast(tf.shape(cls4_score)[0], tf.float32)*tf.cast((1-reject2_f), tf.float32), tf.int32))
      rois = tf.gather(rois, tf.reshape(cls4_inds_2,[-1]))
      fc_combine4_2 = tf.gather(fc_combine4_2, tf.reshape(cls4_inds_2,[-1]))
      cls4_score = tf.gather(cls4_score, tf.reshape(cls4_inds_2,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls4_inds_2,[-1]))


      #self._act_summaries.append(self.endpoint['conv4_3'])

      # #---------------------------------------------------------rcnn 1---------------------------------------------------------------#
      #generate target
      if is_training:          
        with tf.control_dependencies([rpn_labels]):
          roi_scores = tf.gather(roi_scores, tf.reshape(cls4_inds_1,[-1]))
          roi_scores = tf.gather(roi_scores, tf.reshape(cls4_inds_2,[-1]))
      #    rois, _, passinds5 = self._proposal_target_layer(rois, roi_scores, "rpn1_rois", batch1)
      #    cls4_score = tf.gather(cls4_score, tf.reshape(passinds5,[-1]))
      #    cls3_score = tf.gather(cls3_score, tf.reshape(passinds5,[-1]))
      #    fc_combine4_2 = tf.gather(fc_combine4_2, tf.reshape(passinds5,[-1]))

      if cfg.POOLING_MODE == 'crop':
        pool51 = self._crop_pool_layer(self.endpoint['conv5_3'], rois, 16, 7, "pool51")
      else:
        raise NotImplementedError


      pool51_conv = slim.conv2d(pool51, 256, [1, 1], trainable=is_training, weights_initializer=initializer, scope="pool51_conv")
      pool51_avg = slim.avg_pool2d(pool51_conv, [7, 7], padding='SAME', scope='pool51_avg', stride = 1) 
      pool51_flat = slim.flatten(pool51_avg, scope='flatten51') 

      fc5_2 = slim.fully_connected(pool51_flat, 512, scope='fc5_2', weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training)
      
      # if is_training:
      #   fc5_2 = slim.dropout(fc5_2, keep_prob=0.5, is_training=True, scope='fc5_2')

      fc5_2 = self._score_add_up(fc_combine4_2, fc5_2, factor1, factor2, 'fc_52_comb')

      #combine
      scale5_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale5_2')

      fc_combine5_2 = tf.scalar_mul(scale5_2, fc5_2)

      cls5_score = slim.fully_connected(fc_combine5_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls5_score')


      #store RCNN2
      self._predictions["cls1_score"] = cls5_score

      cls5_prob = self._softmax_layer(cls5_score, "cls5_prob")

      #reject via threshold
      cls5_inds_1 = tf.reshape(tf.where(tf.less(cls5_prob[:,0], reject1)), [-1])
      rois = tf.gather(rois, tf.reshape(cls5_inds_1,[-1]))
      cls5_score = tf.gather(cls5_score, tf.reshape(cls5_inds_1,[-1]))
      cls4_score = tf.gather(cls4_score, tf.reshape(cls5_inds_1,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls5_inds_1,[-1]))

      #reject via factor
      _, cls5_inds_2 = tf.nn.top_k(cls5_score[:,0], tf.cast(tf.cast(tf.shape(cls5_score)[0], tf.float32)*tf.cast((1-reject1_f), tf.float32), tf.int32))
      rois = tf.gather(rois, tf.reshape(cls5_inds_2,[-1]))
      cls5_score = tf.gather(cls5_score, tf.reshape(cls5_inds_2,[-1]))
      cls4_score = tf.gather(cls4_score, tf.reshape(cls5_inds_2,[-1]))
      cls3_score = tf.gather(cls3_score, tf.reshape(cls5_inds_2,[-1]))

      #self._act_summaries.append(self.endpoint['conv5_2'])

      #-------------------------------------------------------rcnn -------------------------------------------------------#
      #generate target
      if is_training:             
        with tf.control_dependencies([rpn_labels]):
          roi_scores = tf.gather(roi_scores, tf.reshape(cls5_inds_1,[-1])) 
          roi_scores = tf.gather(roi_scores, tf.reshape(cls5_inds_2,[-1])) 
          rois, _, passinds = self._proposal_target_layer(rois, roi_scores, "rpn_rois", batch)
          cls5_score = tf.gather(cls5_score, tf.reshape(passinds,[-1]))
          cls4_score = tf.gather(cls4_score, tf.reshape(passinds,[-1]))
          cls3_score = tf.gather(cls3_score, tf.reshape(passinds,[-1]))

      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(self.endpoint['conv5_3'], rois, 16, 7, "pool5")
        self.endpoint['pool5'] = pool5
      else:
        raise NotImplementedError

      pool5_flat = slim.flatten(pool5, scope='flatten')
      self._predictions['p5f'] = pool5_flat

      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      cls0_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score_pre')

      self._predictions["cls0_score"] = cls0_score

      # I find seeting up this learnable scale is useless, you can still have train if you want to
      cls3_score_scale = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'cls3_score_scale')
      cls2_score_scale = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'cls2_score_scale')
      cls1_score_scale = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'cls1_score_scale')
      cls0_score_scale = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'cls0_score_scale')

      cls_score = cls3_score*cls3_score_scale*0.25 + cls4_score*cls2_score_scale*0.25 + cls5_score*cls1_score_scale*0.25 + cls0_score*cls0_score_scale*0.25

      # cls_score = cls3_score*0.25 + cls4_score*0.25 + cls5_score*0.25 + cls0_score*0.25

      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                       weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')

      cls_prob = self._softmax_layer(cls_score, "cls_prob")


      self._act_summaries.append(self.endpoint['conv5_3'])
      ###########################################################RCNN DONE############################################################

      #store rpn values
      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      

      #store RCNN
      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob  
      self._predictions["bbox_pred"] = bbox_pred
      self._predictions["rois"] = rois
      #####only for training######


      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'vgg_16/conv1/conv1_1/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv, 
                                      "vgg_16/fc7/weights": fc7_conv,
                                      "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
