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

factor = cfg.SCORE_FACTOR

OHEM4_2 = cfg.TRAIN.OHEM4_2
OHEM4_3 = cfg.TRAIN.OHEM4_3
OHEM5_2 = cfg.TRAIN.OHEM5_2
OHEM5_3 = cfg.TRAIN.OHEM5_3

reject4_3 = cfg.TRAIN.REJECT4_3
reject5_2 = cfg.TRAIN.REJECT5_2
reject5_3 = cfg.TRAIN.REJECT5_3


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
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')


      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      
      #save conv4_3
      self.endpoint['conv4_2'] = net
      net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope = "conv4/conv4_3")

      #save conv4_3
      self.endpoint['conv4_3'] = net

      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      #save conv5_2
      self.endpoint['conv5_2'] = net
      net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope = "conv5/conv5_3")

      #save conv5_3
      self.endpoint['conv5_3'] = net

      self._act_summaries.append(net)
      self._layers['head'] = net




################################################################RPN#############################################################################

      ##-----------------------------------------------rpn 4-2------------------------------------------------------------##
      # rpn 4

      # build the anchors for the image
      self._anchor_component(4)

      rpn4_2 = slim.conv2d(self.endpoint['conv4_2'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn4_2_conv/3x3")
      self._act_summaries.append(rpn4_2)
      rpn4_2_cls_score = slim.conv2d(rpn4_2, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn4_2_cls_score')
      # change it so that the score has 2 as its channel size
      rpn4_2_cls_score_reshape = self._reshape_layer(rpn4_2_cls_score, 2, 'rpn4_2_cls_score_reshape')
      # rpn4_2_cls_prob_reshape = self._softmax_layer(rpn4_2_cls_score_reshape, "rpn4_2_cls_prob_reshape")
      # rpn4_2_cls_prob = self._reshape_layer(rpn4_2_cls_prob_reshape, self._num_anchors * 2, "rpn4_2_cls_prob")

      # #tmp not need: box
      # rpn4_2_bbox_pred = slim.conv2d(rpn4_2, self._num_anchors * 4, [1, 1], trainable=is_training,
      #                             weights_initializer=initializer,
      #                             padding='VALID', activation_fn=None, scope='rpn4_2_bbox_pred')



      if is_training:
        # rois1, roi1_scores = self._proposal_layer(rpn4_2_cls_prob, rpn4_2_bbox_pred, "rois1")
        rpn4_2_labels, rpn4_2_pass_inds, rpn4_2_rej_inds = self._anchor_target_layer(4, rpn4_2_cls_score, "anchor4_2", [], [], OHEM4_2, -1, [])
        # # Try to have a determinestic order for the computing graph, for reproducibility
        # with tf.control_dependencies([rpn4_2_labels]):
        #   rois1, _ = self._proposal_target_layer(rois1, roi1_scores, "rpn4_2_rois")
      # else:
      #   if cfg.TEST.MODE == 'nms':
      #     rois1, _ = self._proposal_layer(rpn4_2_cls_prob, rpn4_2_bbox_pred, "rois1", [])
      #   elif cfg.TEST.MODE == 'top':
      #     rois1, _ = self._proposal_top_layer(rpn4_2_cls_prob, rpn4_2_bbox_pred, "rois1")
      #   else:
      #     raise NotImplementedError

      ##---------------------------------------------rpn 4-2 done------------------------------------------------------------## 

      



      ##-----------------------------------------------rpn 4-3------------------------------------------------------------##
      # rpn 4

      # build the anchors for the image
      self._anchor_component(4)

      rpn4_3 = slim.conv2d(self.endpoint['conv4_3'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn4_3_conv/3x3")
      self._act_summaries.append(rpn4_3)
      rpn4_3_cls_score = slim.conv2d(rpn4_3, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn4_3_cls_score')
      # change it so that the score has 2 as its channel size
      rpn4_3_cls_score_reshape = self._reshape_layer(rpn4_3_cls_score, 2, 'rpn4_3_cls_score_reshape')
      # rpn4_3_cls_prob_reshape = self._softmax_layer(rpn4_3_cls_score_reshape, "rpn4_3_cls_prob_reshape")
      # rpn4_3_cls_prob = self._reshape_layer(rpn4_3_cls_prob_reshape, self._num_anchors * 2, "rpn4_3_cls_prob")

      # #tmp not need box
      # rpn4_3_bbox_pred = slim.conv2d(rpn4_3, self._num_anchors * 4, [1, 1], trainable=is_training,
      #                             weights_initializer=initializer,
      #                             padding='VALID', activation_fn=None, scope='rpn4_3_bbox_pred')

      
      #add up 2 scores rpn4_2 and rpn
      rpn4_cls_score = self._score_add_up(rpn4_3_cls_score, rpn4_2_cls_score, factor, 'rpn4_cls_score')

      #used added up score
      rpn4_cls_score_reshape = self._reshape_layer(rpn4_cls_score, 2, 'rpn4_cls_score_reshape')
      rpn4_cls_prob_reshape = self._softmax_layer(rpn4_cls_score_reshape, "rpn4_cls_prob_reshape")
      rpn4_cls_prob = self._reshape_layer(rpn4_cls_prob_reshape, self._num_anchors * 2, "rpn4_cls_prob")


      if is_training:
        # rois1, roi1_scores = self._proposal_layer(rpn4_3_cls_prob, rpn4_3_bbox_pred, "rois1")
        rpn4_3_labels, rpn4_3_pass_inds, rpn4_3_rej_inds = self._anchor_target_layer(4, rpn4_cls_score, "anchor4_3", rpn4_2_cls_score_reshape, [], OHEM4_3, reject4_3,[])

      #   #generate proposal - using add up score
      #   rois4, roi4_scores = self._proposal_layer(4, rpn4_cls_prob, rpn4_3_bbox_pred, "rois4", rpn4_3_pass_inds, [], rpn4_2_bbox_pred)

      #   # Try to have a determinestic order for the computing graph, for reproducibility
      #   with tf.control_dependencies([rpn4_3_labels]):
      #     rois4, _ = self._proposal_target_layer(rois4, roi4_scores, "rpn4_rois")
      # else:
      #   if cfg.TEST.MODE == 'nms':
      #     # using added up score for proposal
      #     rois4, _ = self._proposal_layer(4, rpn4_cls_prob, rpn4_3_bbox_pred, "rois4", [], rpn4_2_cls_score, rpn4_2_bbox_pred)
      #   elif cfg.TEST.MODE == 'top':
      #     # using added up score for proposal
      #     rois4, _ = self._proposal_top_layer(4, rpn4_cls_prob, rpn4_3_bbox_pred, "rois4", rpn4_2_cls_score, rpn4_2_bbox_pred)
      #   else:
      #     raise NotImplementedError



      ##---------------------------------------------rpn 4-3 done------------------------------------------------------------## 


      #----------------------------------pass rpn4 information to rpn5-------------------------------------------------##
      rpn4_cls_score_resize = slim.avg_pool2d(rpn4_cls_score, [2, 2], padding='SAME', scope='rpn4_cls_score_resize')
      rpn4_cls_score_reshape_resize = self._reshape_layer(rpn4_cls_score, 2, 'rpn4_cls_score_reshape_resize')

      #tmp not need: box
      #rpn4_3_bbox_pred_resize = slim.avg_pool2d(rpn4_3_bbox_pred, [2, 2], padding='SAME', scope='rpn4_3_bbox_pred_resize')

      #------------------------------------------------done-------------------------------------------------------------#

      ##-----------------------------------------------rpn 5-2------------------------------------------------------------##
      # rpn 5-2

      # build the anchors for the image
      self._anchor_component(5)

      rpn5 = slim.conv2d(self.endpoint['conv5_2'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn5_conv/3x3")
      self._act_summaries.append(rpn5)
      rpn5_cls_score = slim.conv2d(rpn5, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn5_cls_score')
      # change it so that the score has 2 as its channel size
      rpn5_cls_score_reshape = self._reshape_layer(rpn5_cls_score, 2, 'rpn5_cls_score_reshape')
      # rpn5_cls_prob_reshape = self._softmax_layer(rpn5_cls_score_reshape, "rpn5_cls_prob_reshape")
      # rpn5_cls_prob = self._reshape_layer(rpn5_cls_prob_reshape, self._num_anchors * 2, "rpn5_cls_prob")
      rpn5_bbox_pred = slim.conv2d(rpn5, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn5_bbox_pred')


      #add up 2 scores rpn4 and rpn5
      rpn45_cls_score = self._score_add_up(rpn4_cls_score_resize, rpn5_cls_score, factor, 'rpn45_cls_score')

      #used added up score
      rpn45_cls_score_reshape = self._reshape_layer(rpn45_cls_score, 2, 'rpn45_cls_score_reshape')
      rpn45_cls_prob_reshape = self._softmax_layer(rpn45_cls_score_reshape, "rpn45_cls_prob_reshape")
      rpn45_cls_prob = self._reshape_layer(rpn45_cls_prob_reshape, self._num_anchors * 2, "rpn45_cls_prob")



      if is_training:
        # rois1, roi1_scores = self._proposal_layer(rpn5_cls_prob, rpn5_bbox_pred, "rois1")
        rpn5_labels, rpn5_pass_inds, rpn5_rej_inds = self._anchor_target_layer(5, rpn45_cls_score, "anchor5", rpn4_cls_score_reshape_resize, [], OHEM5_2, reject5_2, [])
        # # Try to have a determinestic order for the computing graph, for reproducibility
        # with tf.control_dependencies([rpn5_labels]):
        #   rois1, _ = self._proposal_target_layer(rois1, roi1_scores, "rpn5_rois")
      # else:
      #   if cfg.TEST.MODE == 'nms':
      #     rois1, _ = self._proposal_layer(rpn5_cls_prob, rpn5_bbox_pred, "rois1", [])
      #   elif cfg.TEST.MODE == 'top':
      #     rois1, _ = self._proposal_top_layer(rpn5_cls_prob, rpn5_bbox_pred, "rois1")
      #   else:
      #     raise NotImplementedError

      ##---------------------------------------------rpn 5-2 done------------------------------------------------------------##


      ##-----------------------------------------------rpn 5-3-----------------------------------------------------------------##
      self._anchor_component(5)

      rpn = slim.conv2d(self.endpoint['conv5_3'], 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')



      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

      #add up 2 scores rpn5 and rpn
      rpn56_cls_score = self._score_add_up(rpn45_cls_score, rpn_cls_score, factor, 'rpn56_cls_score')

      #used added up score
      rpn56_cls_score_reshape = self._reshape_layer(rpn56_cls_score, 2, 'rpn56_cls_score_reshape')
      rpn56_cls_prob_reshape = self._softmax_layer(rpn56_cls_score_reshape, "rpn56_cls_prob_reshape")
      rpn56_cls_prob = self._reshape_layer(rpn56_cls_prob_reshape, self._num_anchors * 2, "rpn56_cls_prob")


      if is_training:
        #compute anchor loss       
        rpn_labels, rpn_pass_inds, rpn_rej_inds = self._anchor_target_layer(5, rpn56_cls_score, "anchor", rpn45_cls_score_reshape, rpn5_bbox_pred, OHEM5_3, reject5_3, rpn5_rej_inds)

        # #compute anchor loss - using add up score       
        #rpn_labels, rpn_pass_inds = self._anchor_target_layer(6, rpn56_cls_score, "anchor", rpn5_cls_prob_reshape, rpn5_bbox_pred, OHEM2)

        # #generate proposal
        # rois, roi_scores = self._proposal_layer(6, rpn_cls_prob, rpn_bbox_pred, "rois", rpn_pass_inds, [], rpn5_bbox_pred)

        #generate proposal - using add up score
        rois, roi_scores = self._proposal_layer(5, rpn56_cls_prob, rpn_bbox_pred, "rois", rpn_pass_inds, [], rpn5_bbox_pred)

        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          # rois, _ = self._proposal_layer(6, rpn_cls_prob, rpn_bbox_pred, "rois", [], rpn_cls_prob_reshape, rpn5_bbox_pred)
          # using added up score for proposal
          rois, _ = self._proposal_layer(5, rpn56_cls_prob, rpn_bbox_pred, "rois", [], rpn5_cls_score, rpn5_bbox_pred)
        elif cfg.TEST.MODE == 'top':
          # rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", rpn_cls_prob_reshape, rpn5_bbox_pred)
          # using added up score for proposal
          rois, _ = self._proposal_top_layer(5, rpn56_cls_prob, rpn_bbox_pred, "rois", rpn5_cls_score, rpn5_bbox_pred)
        else:
          raise NotImplementedError

      ##-----------------------------------------------rpn 5-3 done--------------------------------------------------------------##


################################################################RPN finish#############################################################################
      
      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net, rois, "pool5")
      else:
        raise NotImplementedError

      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                       weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
      
      #store rpn4_2 values
      self._predictions["rpn4_2_cls_score"] = rpn4_2_cls_score
      self._predictions["rpn4_2_cls_score_reshape"] = rpn4_2_cls_score_reshape
      #self._predictions["rpn4_2_cls_prob"] = rpn4_2_cls_prob
      
      #tmp not need: box
      #self._predictions["rpn4_2_bbox_pred"] = rpn4_2_bbox_pred  


      #store rpn4_3 values
      self._predictions["rpn4_3_cls_score"] = rpn4_3_cls_score
      self._predictions["rpn4_3_cls_score_reshape"] = rpn4_3_cls_score_reshape
      # self._predictions["rpn4_3_cls_prob"] = rpn4_3_cls_prob

      #tmp not need box
      #self._predictions["rpn4_3_bbox_pred"] = rpn4_3_bbox_pred


      #store rpn5 values
      self._predictions["rpn5_cls_score"] = rpn5_cls_score
      self._predictions["rpn5_cls_score_reshape"] = rpn5_cls_score_reshape
      #self._predictions["rpn5_cls_prob"] = rpn5_cls_prob
      self._predictions["rpn5_bbox_pred"] = rpn5_bbox_pred
      #done


      #store rpn values
      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      #done

      #store added up 4 score values
      self._predictions["rpn4_cls_score"] = rpn4_cls_score
      self._predictions["rpn4_cls_score_reshape"] = rpn4_cls_score_reshape
      self._predictions["rpn4_cls_prob"] = rpn4_cls_prob
      #done

      #store added up 45 score values
      self._predictions["rpn45_cls_score"] = rpn45_cls_score
      self._predictions["rpn45_cls_score_reshape"] = rpn45_cls_score_reshape
      self._predictions["rpn45_cls_prob"] = rpn45_cls_prob
      #done

      #store added up 56 score values
      self._predictions["rpn56_cls_score"] = rpn56_cls_score
      self._predictions["rpn56_cls_score_reshape"] = rpn56_cls_score_reshape
      self._predictions["rpn56_cls_prob"] = rpn56_cls_prob
      #done

      #store rpn4-resize values
      self._predictions["rpn4_cls_score_resize"] = rpn4_cls_score_resize
      #self._predictions["rpn4_3_bbox_pred_resize"] = rpn4_3_bbox_pred_resize
      #done

      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._predictions["bbox_pred"] = bbox_pred
      self._predictions["rois"] = rois

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