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

#OHEM
OHEM3 = cfg.TRAIN.OHEM3
OHEM2 = cfg.TRAIN.OHEM2
OHEM1 = cfg.TRAIN.OHEM1
OHEM = cfg.TRAIN.OHEM

#FRCN BATCH
frcn_batch3 = cfg.TRAIN.FRCN_BATCH3
frcn_batch2 = cfg.TRAIN.FRCN_BATCH2
frcn_batch1 = cfg.TRAIN.FRCN_BATCH1
frcn_batch = cfg.TRAIN.FRCN_BATCH

#RPN BATCH
rpn3_batch = cfg.TRAIN.RPN3_BATCH
rpn2_batch = cfg.TRAIN.RPN2_BATCH
rpn1_batch = cfg.TRAIN.RPN1_BATCH
rpn_batch = cfg.TRAIN.RPN_BATCH


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

      self.endpoint['conv3_3'] = net

      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')

      self.endpoint['conv4_3'] = net

      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
      #                   trainable=is_training, scope='conv5')

      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      #store conv5_2
      self.endpoint['conv5_2'] = net

      #continue conv5/conv5_3
      net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope = "conv5/conv5_3")


      #store conv5_3
      self.endpoint['conv5_3'] = net
      # self._predictions['conv5_3'] = net

      # build the anchors for the image
      self._anchor_component()



      #-----------------------------------------------rpn 3------------------------------------------------------------#
      conv3_resize = slim.avg_pool2d(self.endpoint['conv3_3'], [4, 4], padding='SAME', scope='conv3_resize', stride = 4)

      # rpn 3
      rpn3 = slim.conv2d(conv3_resize, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn3_conv/3x3")
      self._act_summaries.append(rpn3)
      rpn3_cls_score = slim.conv2d(rpn3, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn3_cls_score')
      # change it so that the score has 2 as its channel size
      rpn3_cls_score_reshape = self._reshape_layer(rpn3_cls_score, 2, 'rpn3_cls_score_reshape')
      rpn3_cls_prob_reshape = self._softmax_layer(rpn3_cls_score_reshape, "rpn3_cls_prob_reshape")
      rpn3_cls_prob = self._reshape_layer(rpn3_cls_prob_reshape, self._num_anchors * 2, "rpn3_cls_prob")
      rpn3_bbox_pred = slim.conv2d(rpn3, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn3_bbox_pred')

      if is_training:
        # rois1, roi1_scores = self._proposal_layer(rpn1_cls_prob, rpn1_bbox_pred, "rois1")
        rpn3_labels, rpn3_rej_inds = self._anchor_target_layer(rpn3_cls_score, "anchor3", [], [], OHEM3, rpn3_batch)
        # # Try to have a determinestic order for the computing graph, for reproducibility

        rois3, roi3_scores, frcn3_order, frcn3_keep, frcn3_passinds,frcn3_score = self._proposal_layer(rpn3_cls_prob, rpn3_bbox_pred, "rois3", [], [], [], [], [], [], [], [])

        with tf.control_dependencies([rpn3_labels]):
          rois3, _, rois3_keep_inds, frcn3_score = self._proposal_target_layer(rois3, roi3_scores, "rpn3_rois", frcn_batch3, frcn3_score)
      else:
        if cfg.TEST.MODE == 'nms':
          rois3, _,  frcn3_order, frcn3_keep, frcn3_passinds, frcn3_score = self._proposal_layer(rpn3_cls_prob, rpn3_bbox_pred, "rois3", [], [], [], [], [], [], [], [])
        elif cfg.TEST.MODE == 'top':
          rois3, _, frcn3_top_inds, frcn3_passinds, frcn3_score = self._proposal_top_layer(rpn3_cls_prob, rpn3_bbox_pred, "rois3", [], [], [], [] ,[])
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool31 = self._crop_pool_layer(conv3_resize, rois3, "pool31")
      else:
        raise NotImplementedError

      pool31_flat = slim.flatten(pool31, scope='flatten31') 

      fc3_2 = slim.fully_connected(pool31_flat, 512, scope='fc3_2', weights_initializer=initializer)

      #combine
      scale3_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale3_2')
      # self._predictions['scale3_2'] = scale3_2
      fc_combine3_2 = tf.scalar_mul(scale3_2, fc3_2)

      cls3_score = slim.fully_connected(fc_combine3_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls3_score')
      cls3_prob = self._softmax_layer(cls3_score, "cls1_prob")

      self._act_summaries.append(conv3_resize)



      ##-----------------------------------------------rpn 2------------------------------------------------------------##
      conv4_resize = slim.avg_pool2d(self.endpoint['conv4_3'], [2, 2], padding='SAME', scope='conv4_3_resize')
      # rpn 4
      rpn2 = slim.conv2d(conv4_resize, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn2_conv/3x3")

      self._act_summaries.append(rpn2)

      rpn2_cls_score = slim.conv2d(rpn2, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn2_cls_score_pre')

      rpn2_bbox_pred = slim.conv2d(rpn2, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn2_bbox_pred')


      #add up 2 scores rpn1 and rpn
      rpn2_cls_score = self._score_add_up(rpn3_cls_score, rpn2_cls_score, factor1, factor2, 'rpn2_cls_score')

      #used added up score
      rpn2_cls_score_reshape = self._reshape_layer(rpn2_cls_score, 2, 'rpn2_cls_score_reshape')
      rpn2_cls_prob_reshape = self._softmax_layer(rpn2_cls_score_reshape, "rpn2_cls_prob_reshape")
      rpn2_cls_prob = self._reshape_layer(rpn2_cls_prob_reshape, self._num_anchors * 2, "rpn2_cls_prob")

      
      if is_training:
        #compute anchor loss       
        rpn2_labels, rpn2_rej_inds = self._anchor_target_layer(rpn2_cls_score, "anchor2", rpn3_cls_prob_reshape, [], OHEM2, rpn2_batch)

        #generate proposal - using add up score
        # rois2, roi2_scores, frcn2_order, frcn2_keep, frcn2_passinds, frcn2_score = self._proposal_layer(rpn2_cls_prob, rpn2_bbox_pred, "rois2", rpn2_rej_inds, [], [rpn3_bbox_pred], frcn3_order, frcn3_keep, frcn3_passinds, rois3_keep_inds, cls3_score)
        rois2, roi2_scores, frcn2_order, frcn2_keep, frcn2_passinds, frcn2_score = self._proposal_layer(rpn2_cls_prob, rpn2_bbox_pred, "rois2", rpn2_rej_inds, [], [], frcn3_order, frcn3_keep, frcn3_passinds, rois3_keep_inds, cls3_score)

        # # Try to have a determinestic order for the computing graph, for reproducibility
        # with tf.control_dependencies([rpn2_labels]):
        rois2, _, rois2_keep_inds, frcn2_score = self._proposal_target_layer(rois2, roi2_scores, "rpn2_rois", frcn_batch2, frcn2_score)
      
      else:
        if cfg.TEST.MODE == 'nms':
          # rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], rpn_cls_prob_reshape, rpn1_bbox_pred)
          # using added up score for proposal
          rois2, _, frcn2_order, frcn2_keep, frcn2_passinds, frcn2_score = self._proposal_layer(rpn2_cls_prob, rpn2_bbox_pred, "rois2", [], rpn3_cls_prob, [rpn3_bbox_pred], frcn3_order, frcn3_keep, frcn3_passinds, [], cls3_score)
        elif cfg.TEST.MODE == 'top':
          # rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn_cls_prob_reshape, rpn1_bbox_pred)
          # using added up score for proposal
          rois2, _, frcn2_top_inds, frcn2_passinds, frcn2_score = self._proposal_top_layer(rpn2_cls_prob, rpn2_bbox_pred, "rois", rpn3_cls_prob, [], frcn3_top_inds, frcn3_passinds, cls3_score)
        else:
          raise NotImplementedError

           # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool41 = self._crop_pool_layer(conv4_resize, rois2, "pool41")
      else:
        raise NotImplementedError

      pool41_flat = slim.flatten(pool41, scope='flatten41') 

      fc4_2 = slim.fully_connected(pool41_flat, 512, scope='fc4_2', weights_initializer=initializer)

      #combine
      scale4_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale4_2')
      # self._predictions['scale3_2'] = scale3_2
      fc_combine4_2 = tf.scalar_mul(scale4_2, fc4_2)

      cls2_score = slim.fully_connected(fc_combine4_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls3_score_pre')

      cls2_score = self._score_add_up(frcn2_score, cls2_score, factor1, factor2, 'cls2_score')

      cls2_prob = self._softmax_layer(cls2_score, "cls2_prob")

      self._act_summaries.append(conv4_resize)


      ##-----------------------------------------------rpn 1------------------------------------------------------------##
      # self._predictions['conv3'] = conv3_resize


      # rpn 1
      rpn1 = slim.conv2d(self.endpoint['conv5_2'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn1_conv/3x3")
      self._act_summaries.append(rpn1)
      rpn1_cls_score = slim.conv2d(rpn1, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn1_cls_score')
      # change it so that the score has 2 as its channel size
      rpn1_cls_score_reshape = self._reshape_layer(rpn1_cls_score, 2, 'rpn1_cls_score_reshape')
      rpn1_cls_prob_reshape = self._softmax_layer(rpn1_cls_score_reshape, "rpn1_cls_prob_reshape")
      rpn1_cls_prob = self._reshape_layer(rpn1_cls_prob_reshape, self._num_anchors * 2, "rpn1_cls_prob")
      rpn1_bbox_pred = slim.conv2d(rpn1, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn1_bbox_pred')

      if is_training:
        #compute anchor loss       
        rpn1_labels, rpn1_rej_inds = self._anchor_target_layer(rpn1_cls_score, "anchor1", rpn2_cls_prob_reshape, [rpn3_bbox_pred, rpn2_bbox_pred], OHEM1, rpn1_batch)

        #generate proposal - using add up score
        rois1, roi1_scores, frcn1_order, frcn1_keep, frcn1_passinds, frcn1_score = self._proposal_layer(rpn1_cls_prob, rpn1_bbox_pred, "rois1", rpn1_rej_inds, [], [rpn3_bbox_pred, rpn2_bbox_pred], frcn2_order, frcn2_keep, frcn2_passinds, rois2_keep_inds, cls2_score)

        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn1_labels]):
          rois1, _, rois1_keep_inds, frcn1_score = self._proposal_target_layer(rois1, roi1_scores, "rpn1_rois", frcn_batch1, frcn1_score)
      else:
        if cfg.TEST.MODE == 'nms':
          rois1, _, frcn1_order, frcn1_keep, frcn1_passinds, frcn1_score = self._proposal_layer(rpn1_cls_prob, rpn1_bbox_pred, "rois1", [], rpn2_cls_prob, [rpn3_bbox_pred, rpn2_bbox_pred], frcn2_order, frcn2_keep, frcn2_passinds, [], cls2_score)
        
        elif cfg.TEST.MODE == 'top':

          rois1, _, frcn1_top_inds, frcn1_passinds, frcn1_score = self._proposal_top_layer(rpn1_cls_prob, rpn1_bbox_pred, "rois1", rpn1_cls_prob, [rpn3_bbox_pred, rpn2_bbox_pred], frcn2_top_inds, frcn2_passinds, cls2_score)
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool51 = self._crop_pool_layer(self.endpoint['conv5_2'], rois1, "pool51")
      else:
        raise NotImplementedError

      pool51_flat = slim.flatten(pool51, scope='flatten1') 

      fc5_2 = slim.fully_connected(pool51_flat, 512, scope='fc5_2', weights_initializer=initializer)

      #combine
      scale5_2 = tf.Variable(tf.cast(1, tf.float32), trainable = is_training, name = 'scale5_2')
      self._predictions['scale5_2'] = scale5_2
      fc_combine5_2 = tf.scalar_mul(scale5_2, fc5_2)

      cls1_score = slim.fully_connected(fc_combine5_2, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls1_score')
      cls1_prob = self._softmax_layer(cls1_score, "cls1_prob")

      self._act_summaries.append(self.endpoint['conv5_2'])
      ##---------------------------------------------rpn 1 done------------------------------------------------------------##

      
      
      ##-----------------------------------------------rpn-----------------------------------------------------------------##
      rpn = slim.conv2d(self.endpoint['conv5_3'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)

      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score_pre')

      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

      #add up 2 scores rpn1 and rpn
      rpn_cls_score = self._score_add_up(rpn1_cls_score, rpn_cls_score, factor1, factor2, 'rpn_cls_score')

      #used added up score
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")


      if is_training:
        #compute anchor loss       
        rpn_labels, rpn_rej_inds = self._anchor_target_layer(rpn_cls_score, "anchor", rpn1_cls_prob_reshape, [rpn3_bbox_pred, rpn2_bbox_pred, rpn1_bbox_pred], OHEM, rpn_batch)

        # #compute anchor loss - using add up score       
        #rpn_labels, rpn_pass_inds = self._anchor_target_layer(rpn12_cls_score, "anchor", rpn1_cls_prob_reshape, rpn1_bbox_pred, OHEM2)

        # #generate proposal
        # rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn_pass_inds, [], rpn1_bbox_pred)

        #generate proposal - using add up score
        rois, roi_scores, frcn_order, frcn_keep, frcn_passinds, frcn_score = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn_rej_inds, [], [rpn3_bbox_pred, rpn2_bbox_pred, rpn1_bbox_pred], frcn1_order, frcn1_keep, frcn1_passinds, rois1_keep_inds, cls1_score)

        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _, rois_keep_inds, frcn_score = self._proposal_target_layer(rois, roi_scores, "rpn_rois", frcn_batch, frcn_score)
      else:
        if cfg.TEST.MODE == 'nms':
          # rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], rpn_cls_prob_reshape, rpn1_bbox_pred)
          # using added up score for proposal
          rois, _, frcn_order, frcn_keep, frcn_passinds, frcn_score = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", [], rpn1_cls_prob, [rpn3_bbox_pred, rpn2_bbox_pred, rpn1_bbox_pred], frcn1_order, frcn1_keep, frcn1_passinds, [], cls1_score)
        elif cfg.TEST.MODE == 'top':
          # rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn_cls_prob_reshape, rpn1_bbox_pred)
          # using added up score for proposal
          rois, _, frcn_top_inds, frcn_passinds, frcn_score = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn1_cls_prob, [rpn3_bbox_pred, rpn2_bbox_pred, rpn1_bbox_pred], frcn1_top_inds, frcn1_passinds, cls1_score)
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(self.endpoint['conv5_3'], rois, "pool5")
        self.endpoint['pool5'] = pool5
      else:
        raise NotImplementedError

      pool5_flat = slim.flatten(pool5, scope='flatten') 
      self.endpoint['p5f'] = pool5_flat

      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score_pre')


       #add up 2 scores rpn1 and rpn
      cls_score = self._score_add_up(frcn_score, cls_score, factor1, factor2, 'cls_score')

      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                       weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
      

      #keep vgg network
      self._act_summaries.append(self.endpoint['conv5_3'])
      self._layers['head'] = self.endpoint['conv5_3']


      #debug
      #tmp = net['conv2/conv2_2']
      #self._predictions['tmp'] = tmp

      #store rpn3 values
      self._predictions["rpn3_cls_score"] = rpn3_cls_score
      self._predictions["rpn3_cls_score_reshape"] = rpn3_cls_score_reshape
      self._predictions["rpn3_cls_prob"] = rpn3_cls_prob
      self._predictions["rpn3_bbox_pred"] = rpn3_bbox_pred
      #done

      #store rpn3 values
      self._predictions["rpn2_cls_score"] = rpn2_cls_score
      self._predictions["rpn2_cls_score_reshape"] = rpn2_cls_score_reshape
      self._predictions["rpn2_cls_prob"] = rpn2_cls_prob
      self._predictions["rpn2_bbox_pred"] = rpn2_bbox_pred
      #done

      #store rpn1 values
      self._predictions["rpn1_cls_score"] = rpn1_cls_score
      self._predictions["rpn1_cls_score_reshape"] = rpn1_cls_score_reshape
      self._predictions["rpn1_cls_prob"] = rpn1_cls_prob
      self._predictions["rpn1_bbox_pred"] = rpn1_bbox_pred
      #done

      #store rpn values
      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      #done

      #new fc 3
      self._predictions["cls3_score"] = cls3_score
      self._predictions["cls3_prob"] = cls3_prob
      self._predictions["rois3"] = rois3
      #

      #new fc 2
      self._predictions["cls2_score"] = cls2_score
      self._predictions["cls2_prob"] = cls2_prob
      self._predictions["rois2"] = rois2

      #new fc
      self._predictions["cls1_score"] = cls1_score
      self._predictions["cls1_prob"] = cls1_prob
      self._predictions["rois1"] = rois1
      #

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