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
import tensorflow as tf

reject_factor = cfg.TEST.REJECT
boxChain = cfg.BOX_CHAIN
train_reject_factor = cfg.TRAIN.REJECT
frcn_reject = cfg.FRCN_REJCECT

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors, rej_inds, pre_rpn_cls_prob_reshape, pre_bbox_pred, name, pre_order, pre_keep, pre_passinds, target_keeps, pre_frcn_cls_score):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """

  #print(name, "In here")

  passinds = np.asarray([], dtype = np.int32)
  order = np.asarray([])
  keep = []

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

  #print(name + ' before reject postive score ', np.where(scores < 0)[0])


#######################################CASCADE VIA RPN#################################################
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
  if cfg_key == 'TRAIN' and rej_inds.size != 0:

    rejinds = rej_inds
    rejinds.sort()

    #print('before reject ', scores.size)

    # proposals = proposals[passinds, :]
    # scores = scores[passinds]

    # FRCN reject
    scores[rejinds] = -1
    #print(name + ' after reject negtive score ', np.where(scores < 0)[0])

    #print(name + ' pass index', passinds.size)
    #print('after reject ', scores.size)

  #------------------reject done-----------------#

  #--------------------------TEST Reject------------------------------#
  if cfg_key == 'TEST' and pre_rpn_cls_prob_reshape.size != 0:

      # #combine SCORE
      pre_rpn_cls_prob_reshape = np.transpose(pre_rpn_cls_prob_reshape,[0,3,1,2])
      pre_scores = pre_rpn_cls_prob_reshape[:,:num_anchors, :, :]
      pre_scores = pre_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
         
      #reject via factor:
      reject_number = int(len(pre_scores)*reject_factor)

      #set up pass index
      pre_scores = pre_scores.ravel()
      rpn_rejinds = pre_scores.argsort()[::-1][:reject_number]

      # #in case cuda error occur
      # if rpn_passinds is None or rpn_passinds.size == 0:          
      #   rpn_passinds = np.array([0])
      #   print(rpn_passinds)

      rpn_rejinds.sort()

      #reject here
      scores[rpn_rejinds] = -1
  #-------------------------------done---------------------------------#


###################################################################################################



######################################CASCADE VIA FRCN############################################
  if pre_order.size != 0:
    proposals = proposals[pre_order, :]
    scores = scores[pre_order]

  if pre_keep.size != 0:
    proposals = proposals[pre_keep, :]
    scores = scores[pre_keep]

  if pre_passinds.size != 0:
    proposals = proposals[pre_passinds, :]
    scores = scores[pre_passinds]

  if cfg_key == 'TRAIN' and target_keeps.size != 0:
    proposals = proposals[target_keeps, :]
    scores = scores[target_keeps]
##################################################################################################

  
  #####################reject via frcn##############
  if pre_frcn_cls_score.size != 0:
    neg_frcn_cls_score = pre_frcn_cls_score[:, 0]


    frcn_reject_number = int(len(neg_frcn_cls_score)*frcn_reject)

    neg_frcn_cls_score = neg_frcn_cls_score.ravel()
    frcn_rejinds = neg_frcn_cls_score.argsort()[::-1][:frcn_reject_number]

    
    scores[frcn_rejinds] = -1

  ##################################################


  #print('proposal ', proposals)

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  #print(name + ' order ', order.size)

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]


  passinds = np.where(scores != -1)[0]
  #reject via frcn and rpn
  proposals = proposals[passinds]
  scores = scores[passinds]

  #print(name + ' keep ', len(keep))

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

 #-------cls score reshape---------#
  # #pass ccinfo to next FRCN
  # ccInfo = {}
  # ccInfo['passinds'] = passinds
  # ccInfo['order'] = order
  # ccInfo['keep'] = keep

  frcn_cls_score = pre_frcn_cls_score

  #print(name, ' frcn scores ', frcn_cls_score.shape)

  if frcn_cls_score.size != 0:   
    #pass
    frcn_cls_score = pre_frcn_cls_score
    frcn_cls_score = frcn_cls_score[order]
    frcn_cls_score = frcn_cls_score[keep]
    frcn_cls_score = frcn_cls_score[passinds]

  return blob, scores, order, keep, passinds, frcn_cls_score 
