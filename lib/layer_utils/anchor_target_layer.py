# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform
from model.bbox_transform import bbox_transform_inv, clip_boxes

boxChain = cfg.BOX_CHAIN

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors, pre_rpn_cls_prob, pre_bbox_pred, OHEM, reject, rej_inds, name, batch):
  """Same as the anchor target layer in original Fast/er RCNN """

  # DEBUG:
  # print('SCORE ',rpn_cls_score[0][0][0][0])
  # print(all_anchors[0])

  #print('In this')
  # if pre_rpn_cls_prob.size != 0:
  #   print pre_rpn_cls_prob.size

  #print(all_anchors)

  A = num_anchors
  total_anchors = all_anchors.shape[0]
  K = total_anchors / num_anchors
  im_info = im_info[0]

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]


  ##-----------------box gression add up----------------##
  if pre_bbox_pred.size != 0 and boxChain == True:

      #chris: preprocess box_pred
      pre_bbox_pred = np.transpose(pre_bbox_pred,[0,3,1,2])
      bbox_deltas = pre_bbox_pred
      bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
      #chris

      #chris: use previous layer
      proposals = bbox_transform_inv(all_anchors, bbox_deltas)
      #chris
   
      all_anchors = proposals

      #print('anchors' ,all_anchors)
   
  ##-----------------------done-------------------------##


  # only keep anchors inside the image
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  argmax_overlaps = overlaps.argmax(axis=1)
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  gt_argmax_overlaps = overlaps.argmax(axis=0)
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1

  # fg label: above threshold IOU
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  
  ###------------------------reject process---------------------------###
  if pre_rpn_cls_prob.size != 0:

    if name == "anchor5":

      # print(pre_rpn_cls_prob)

      for i in [0,1]:
        reject_factor = reject[i]

        pre_rpn_cls_prob_reshape = np.transpose(pre_rpn_cls_prob[i],[0,3,1,2])
        pre_scores = pre_rpn_cls_prob_reshape[:,:A, :, :]
        pre_scores = pre_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        pre_scores = pre_scores[inds_inside]
            
        neg_reject_number = int(len(inds_inside)*reject_factor*0.75) 
        pos_reject_number = int(len(inds_inside)*reject_factor*0.25)

        pre_scores = pre_scores.ravel()

        neg_rejinds = pre_scores.argsort()[::-1][:neg_reject_number]
        pos_rejinds = pre_scores.argsort()[:pos_reject_number]

        rejinds = np.concatenate((pos_rejinds, neg_rejinds), axis=0)
        labels[rejinds] = -2

        #print(i , ' ', name , ' reject : ', len(np.where(labels == -2)[0]), ' anchors' )
        #print(i , ' ', name , ' reject : ', rejinds, ' anchors' )

    else:

      # print(pre_rpn_cls_prob)
      reject_factor = reject

      pre_rpn_cls_prob_reshape = np.transpose(pre_rpn_cls_prob,[0,3,1,2])
      pre_scores = pre_rpn_cls_prob_reshape[:,:A, :, :]
      pre_scores = pre_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
      
      #print(pre_scores)


      pre_scores = pre_scores[inds_inside]
          
      neg_reject_number = int(len(inds_inside)*reject_factor*0.9)
      pos_reject_number = int(len(inds_inside)*reject_factor*0.1)

      pre_scores = pre_scores.ravel()

      neg_rejinds = pre_scores.argsort()[::-1][:neg_reject_number]
      pos_rejinds = pre_scores.argsort()[:pos_reject_number]
        
      rejinds = np.concatenate((pos_rejinds, neg_rejinds), axis=0)
      labels[rejinds] = -2

  
  ###-------------------------reject done-----------------------------###


  #OHEM
  if OHEM == True:
    # subsample positive labels if we have too many
    # num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    
    num_fg = int(batch)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
      disable_inds = npr.choice(
        fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1

    # subsample negative labels if we have too many
    #num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)

    num_bg = len(fg_inds)*3
    #in case nothing return
    if num_bg < 100:
      num_bg = num_fg*3

    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
      disable_inds = npr.choice(
        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      labels[disable_inds] = -1
      

  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)


  #combine reject index
  if rej_inds.size != 0:
    # print(name, ' prerej ', rej_inds)
    # print(name, ' rej', rejinds)

    # rejinds = np.concatenate((rej_inds, rejinds), axis=0)
    # rejinds = np.unique(rejinds)

    labels[rej_inds] = -2

  #get reject inds after umap   
  final_pass_inds = np.where(labels != -2)[0]
  final_rej_inds = np.where(labels == -2)[0]

  #print(name, 'final_pass_inds', final_pass_inds)
  #print(name, 'final_rej_inds', final_rej_inds)

  # #debug
  # print(name , ' reject : ', len(np.where(labels == -2)[0]), ' anchors' )
  # print(name , ' positive : ', len(np.where(labels == 1)[0]), ' anchors' )
  # print(name , ' negative : ', len(np.where(labels == 0)[0]), ' anchors' )
  # print(name , ' ignores : ', len(np.where(labels == -1)[0]), ' anchors' )

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, final_pass_inds, final_rej_inds


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
