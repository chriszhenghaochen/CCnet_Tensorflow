# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb


#import tensorflow as tf

DEBUG = False
#pass_threshold = 0.3
reject_factor = 0.5
#reject_number = 600

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, pre_rpn_cls_prob_reshape, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """


    # input_shape = tf.shape(rpn_cls_score)
    # if name == 'rpn_cls_prob_reshape':
    #     return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
    #             int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    # #chris
    # if pre_rpn_cls_prob_reshape != []:
    #     print 'rpn1_cls_prob_reshape'
    #     print pre_rpn_cls_prob_reshape.shape

    # #chris




    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap


    
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # #chris
    # print 'height ', height
    # print 'width ',width
    # #chris

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors


    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # # #chris -get number of anchors
    # print 'score shape' , rpn_cls_score.shape
    # print 'reshape ', pre_rpn_cls_prob_reshape.shape
    # print 'number of anchors ' ,  total_anchors

    #print 'RPN ', total_anchors 

    # #chris
    # print 'Score'
    # print rpn_cls_score.shape
    # #print rpn_cls_score
    # #chris

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]


    #

    # #chris
    # print 'total_anchors ', total_anchors
    # print 'inds_inside ', len(inds_inside)
    # #chris

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)


    #chris: in case cuda error here:
    if inds_inside.size == 0:
        print 'Over Reject here'
        inds_inside = np.array([0])

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    #print 'RPN reject ', total_anchors - len(inds_inside)

    # # chris - Debug
    # if pre_rpn_cls_prob_reshape != []:
    #     print '2'

    # else:
    #     print '1'

    # print len(inds_inside)
    # if len(inds_inside) == len(anchors):
    #     print 'TRUE'

    # else:
    #     print 'FALSE'
    #chris




    # # #chris
    # if pre_rpn_cls_prob_reshape != []:
    #     print 'Anchor'
    #     print anchors.shape
    # # print anchors
    # #chris


    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # #chris
    # print 'label shape'
    # print labels.shape
    # print 'labels'
    # print labels
    # print 'anchors'
    # print anchors
    # print anchors.shape

    # #chris

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
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


    # #chris
    # if(pre_rpn_cls_prob_reshape != []):
    #     print 'REJECT'
    # else:
    #     print 'NO REJECT'



    #--------------------------------------------------reject here----------------------------------------#
    if pre_rpn_cls_prob_reshape != []:

        # #chris
        # print 'All Anchor Before Reject'
        # print all_anchors.shape
        # #print all_anchors
        #chris


        pre_rpn_cls_prob_reshape = np.transpose(pre_rpn_cls_prob_reshape,[0,3,1,2])
        pre_scores = pre_rpn_cls_prob_reshape[:,:_num_anchors, :, :]
        pre_scores = pre_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        

        ##---------------------------reject via set label to be -1 ---------------------#
        #do this before reject, wipe out the over boundary
        pre_scores = pre_scores[inds_inside]
        #done 


        #rejct via threshold
        # passinds = np.where(pre_scores > pass_threshold)[0]

        #reject via number
        #print 'pre score ', pre_scores
        #print 'size of pre score ', len(pre_scores)


        ###import: reject via factor####
        reject_number = int(len(inds_inside)*reject_factor)

        pre_scores = pre_scores.ravel()
        rejinds = pre_scores.argsort()[::-1][:reject_number]


        #get threshold
        # pass_threshold = min(pre_scores[rejinds])
        
        #print pass_threshold
        

        # chris
        # set reject samples = -1
        #a = len(np.where(labels == -1)[0]) 

        #important: reject label will be -2
        labels[rejinds] = -2

        #b = len(np.where(labels == -1)[0]) 

        #print 'Reject', b - a, 'samples'
        #chris


        #print 'reject index', rejinds
        ##----------------------------------- done ------------------------------------#

    #--------------------------------------------------reject done----------------------------------------#

    # #chris
    # print 'after reject positive ', len(np.where(labels == 1)[0])
    # print 'after reject negative ', len(np.where(labels == 0)[0])    


    # print 'init positive ', len(np.where(labels == 1)[0])
    # print 'init negative ', len(np.where(labels == 0)[0])




    # #-----------------------------chris: this is a hard core focal loss!!------------------------------------#
    # scores = np.transpose(rpn_cls_score,[0,3,1,2])
    # scores = scores[:,_num_anchors:, :, :]
    # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # scores = scores[inds_inside]

    # pos_scores = np.copy(scores)
    # neg_scores = np.copy(scores)


    # # print 'max ', max(scores)
    # # print 'min ', min(scores)

    # maxscore = max(scores)
    # minscore = min(scores)

    # # #process postive score
    # pos_scores[np.where(labels == -1)[0]] = maxscore
    # pos_scores[np.where(labels == -2)[0]] = maxscore
    # pos_scores[np.where(labels == 0)[0]] = maxscore


    # # #process negtive score
    # neg_scores[np.where(labels == -1)[0]]= minscore
    # neg_scores[np.where(labels == -2)[0]] = minscore
    # neg_scores[np.where(labels == 1)[0]] = minscore

    # # print scores
    # # print pos_scores
    # # print neg_scores



    # #-------------------------------------------------chris---------------------------------------------------#



    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]


    # #this is original way -> random

    #set this for restrict
    if pre_rpn_cls_prob_reshape != []:
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

    # #chris: new way - hard core focal loss
    # if len(fg_inds) > num_fg:
    #     disable_len = len(fg_inds) - num_fg
    #     disable_inds = pos_scores.ravel().argsort()[:disable_len]
    #     labels[disable_inds] = -1


    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]

    # #this is original way -> random
        #set this for restrict
    if pre_rpn_cls_prob_reshape != []:
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)

            #print disable_inds
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
    
    # #chris: new way - hard core focal loss
    # if len(bg_inds) > num_bg:
    #     disable_len = len(bg_inds) - num_bg
    #     disable_inds = neg_scores.ravel().argsort()[::-1][:disable_len]

    #     # print disable_inds
    #     # print neg_scores[disable_inds]

    #     labels[disable_inds] = -1


    #####################

    

    # print 'labels'
    # #print labels
    # print labels.shape

    # print 'score'
    # #print rpn_cls_score
    # print rpn_cls_score.shape
    # #chris

    
    
    # #chris   
    # print 'BATCH_SIZE: '
    # print 'num_fg' , num_fg
    # print 'num_bg' , num_bg
    # print 'final positive ', len(np.where(labels == 1)[0])
    # print 'final negative ', len(np.where(labels == 0)[0])
    # print '\n'
    # print '\n'
    # #chris
    #####################



    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
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

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)


    #chris: get reject inds after umap   
    final_pass_inds = np.where(labels != -2)[0]

    #print total_anchors
    #print len(final_pass_inds)
    #chris: done



    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)

    # #chris
    # print labels
    # #chris
    
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    # #chris
    # print rpn_labels
    # #chris

    # #chris
    # print 'labels'
    # print labels.shape
    # print 'target'
    # print rpn_bbox_targets.shape

    # #chris


    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights, final_pass_inds  #,all_anchors.astype(np.float)



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
