# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

reject_factor = cfg.TEST.REJECT
boxChain = cfg.BOX_CHAIN

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors, pass_inds, pre_rpn_cls_prob_reshape, pre_bbox_pred):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """

  # #debug
  # print('prob = ', rpn_cls_prob)
  # print('pass inds', pass_inds)


  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  im_info = im_info[0]
  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))

  ##----------------------------------chris: regression add up-----------------------------------##
  if pre_bbox_pred.size != 0 and boxChain == True:


      #chris: preprocess box_pred
      pre_bbox_pred = np.transpose(pre_bbox_pred,[0,3,1,2])
      pre_bbox_pred = pre_bbox_pred.transpose((0, 2, 3, 1)).reshape((-1, 4))
      #chris

      # print('anchors 1 ', anchors)

      #chris: use previous layer
      anchors = bbox_transform_inv(anchors, pre_bbox_pred)
      #chris

      #print('anchors 2 ', anchors) 
  ##----------------------------------------chris------------------------------------------------##


  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])


  #--------------TRAIN reject here---------------#
  if cfg_key == 'TRAIN' and pass_inds.size != 0:

    passinds = pass_inds
    passinds.sort()

    #print('before reject ', scores.size)

    proposals = proposals[passinds, :]
    scores = scores[passinds]

    #print('after reject ', scores.size)

    # print(passinds)

  #------------------reject done-----------------#

  # print('after reject',scores.size)

  #--------------------------TEST Reject------------------------------#
  if cfg_key == 'TEST' and pre_rpn_cls_prob_reshape.size != 0:

      # #combine SCORE
      pre_rpn_cls_prob_reshape = np.transpose(pre_rpn_cls_prob_reshape,[0,3,1,2])
      pre_scores = pre_rpn_cls_prob_reshape[:, num_anchors:, :, :]
      pre_scores = pre_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
         
      #reject via factor:
      reject_number = int(len(pre_scores)*reject_factor)


      #set up pass number
      passnumber = len(pre_scores) - reject_number

      if passnumber <= 0:
         passnumber = 1

      #set up pass index
      pre_scores = pre_scores.ravel()
      passinds = pre_scores.argsort()[::-1][:passnumber]

      #in case cuda error occur
      if passinds is None or passinds.size == 0:          
        passinds = np.array([0])
        #print(passinds)

      passinds.sort()

      #reject here
      proposals = proposals[passinds, :]
      scores = scores[passinds]
  #-------------------------------done---------------------------------#


  #print('proposal ',proposals)


  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  #print('scores ', scores)

  return blob, scores
