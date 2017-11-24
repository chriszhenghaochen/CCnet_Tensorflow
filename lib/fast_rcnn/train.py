# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import matplotlib.pyplot as plt
from fast_rcnn.focal_loss import SigmoidFocalClassificationLoss as fl
from tensorflow.python.ops import array_ops
from config import cfg

# pass_threshold = 0.3
# printRecall = False

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        
        # chris: initialize the focal loss
        self.fl = fl(alpha=0.25)
        #done

        # For checkpoint
        self.saver = saver


    #chris 
    #this is for reject:

    def recall1(self, neg_inds, label):
        label = tf.reshape(tf.gather(label, tf.reshape(neg_inds,[-1])),[-1])
        label = tf.reshape(tf.gather(label,tf.where(tf.not_equal(label,-1))),[-1])

        return label


    def recall2(self, neg_inds, label):
        label = tf.reshape(tf.gather(label, tf.reshape(neg_inds,[-1])),[-1])
        true_reject = tf.reshape(tf.gather(label,tf.where(tf.equal(label,0))),[-1])

        return true_reject


    def passSample(self, probs, inds = None, threshold = 0.5):
        new_inds = tf.where(tf.greater(probs, threshold))

        if inds == None:
        # print new_inds
        # inds = tf.concat([inds, new_inds], axis = 0)
            return new_inds

        inds = tf.concat([inds, new_inds], axis = 0)
        
        return inds


    def rejectSample(self, probs, inds = None, threshold = 0.5):
        new_inds = tf.where(tf.less(probs, threshold))

        if inds == None:
        # print new_inds
        # inds = tf.concat([inds, new_inds], axis = 0)
            return new_inds

        inds = tf.concat([inds, new_inds], axis = 0)
        
        return inds


    def reject(self, scores):
        return tf.gather(scores, tf.where(tf.less(scores[:,0], scores[:,1])))[:,0,:]

    #chris


    #------------------------------chris: Focal Loss------------------------------------#
    def focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
        """Compute focal loss for predictions.
                Multi-labels Focal loss formula:
                    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                         ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
            Args:
             prediction_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
             target_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
             weights: A float tensor of shape [batch_size, num_anchors]
             alpha: A scalar tensor for focal loss alpha hyper-parameter
             gamma: A scalar tensor for focal loss gamma hyper-parameter
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_mean(per_entry_cross_ent)


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # #chris
        # # RPN-0
        # # # classification loss
        # rpn0_cls_score = tf.reshape(self.net.get_output('rpn0_cls_score_reshape'),[-1,2])
        # rpn0_label = tf.reshape(self.net.get_output('rpn0-data')[0],[-1])
        # rpn0_cls_score = tf.reshape(tf.gather(rpn0_cls_score,tf.where(tf.not_equal(rpn0_label,-1))),[-1,2])
        # rpn0_label = tf.reshape(tf.gather(rpn0_label,tf.where(tf.not_equal(rpn0_label,-1))),[-1])

        # #chris: Regular loss
        # #rpn1_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn1_cls_score, labels=rpn1_label))
        # #chris: Done

        # #chris: Focal Loss
        # length = tf.size(rpn0_label)
        # # #softmax
        # #rpn1_cls_score = tf.nn.softmax(rpn1_cls_score)
        # rpn0_label = tf.one_hot(indices = rpn0_label, depth=2, on_value=1, off_value=0, axis=-1)
        # rpn0_label = tf.cast(rpn0_label, tf.float32)
        # rpn0_weights = tf.ones([1, length], tf.float32)
        # rpn0_cross_entropy = self.fl.compute_loss(prediction_tensor = rpn0_cls_score, target_tensor = rpn0_label, weights = rpn0_weights)
        # #rpn1_cross_entropy = self.focal_loss(prediction_tensor = rpn1_cls_score, target_tensor = rpn1_label, alpha = 0.5)
        # #chris: Done

        # # # bounding box regression L1 loss
        # # rpn1_bbox_pred = self.net.get_output('rpn1_bbox_pred')
        # # rpn1_bbox_targets = tf.transpose(self.net.get_output('rpn1-data')[1],[0,2,3,1])
        # # rpn1_bbox_inside_weights = tf.transpose(self.net.get_output('rpn1-data')[2],[0,2,3,1])
        # # rpn1_bbox_outside_weights = tf.transpose(self.net.get_output('rpn1-data')[3],[0,2,3,1])

        # # rpn1_smooth_l1 = self._modified_smooth_l1(3.0, rpn1_bbox_pred, rpn1_bbox_targets, rpn1_bbox_inside_weights, rpn1_bbox_outside_weights)
        # # rpn1_loss_box = tf.reduce_mean(tf.reduce_sum(rpn1_smooth_l1, reduction_indices=[1, 2, 3]))
        # # #chris
        # # chris





        ##--------------------------------------------RPN-1-----------------------------------------------------------##
        # # classification loss
        rpn1_cls_score = tf.reshape(self.net.get_output('rpn1_cls_score_reshape'),[-1,2])
        # rpn1_cls_score = tf.reshape(self.net.get_output('rpn01_cls_score_reshape'),[-1,2])
        rpn1_label = tf.reshape(self.net.get_output('rpn1-data')[0],[-1])
        rpn1_cls_score = tf.reshape(tf.gather(rpn1_cls_score,tf.where(tf.not_equal(rpn1_label,-1))),[-1,2])
        rpn1_label = tf.reshape(tf.gather(rpn1_label,tf.where(tf.not_equal(rpn1_label,-1))),[-1])

        #chris: Regular loss
        #rpn1_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn1_cls_score, labels=rpn1_label))
        #chris: Done

        #chris: Focal Loss
        length = tf.size(rpn1_label)
        rpn1_label = tf.one_hot(indices = rpn1_label, depth=2, on_value=1, off_value=0, axis=-1)
        rpn1_label = tf.cast(rpn1_label, tf.float32)
        rpn1_weights = tf.ones([1, length], tf.float32)
        rpn1_cross_entropy = self.fl.compute_loss(prediction_tensor = rpn1_cls_score, target_tensor = rpn1_label, weights = rpn1_weights)
        #rpn1_cross_entropy = self.focal_loss(prediction_tensor = rpn1_cls_score, target_tensor = rpn1_label, alpha = 0.5)
        #chris: Done


        # bounding box regression L1 loss
        rpn1_bbox_pred = self.net.get_output('rpn1_bbox_pred')
        rpn1_bbox_targets = tf.transpose(self.net.get_output('rpn1-data')[1],[0,2,3,1])
        rpn1_bbox_inside_weights = tf.transpose(self.net.get_output('rpn1-data')[2],[0,2,3,1])
        rpn1_bbox_outside_weights = tf.transpose(self.net.get_output('rpn1-data')[3],[0,2,3,1])

        rpn1_smooth_l1 = self._modified_smooth_l1(3.0, rpn1_bbox_pred, rpn1_bbox_targets, rpn1_bbox_inside_weights, rpn1_bbox_outside_weights)
        rpn1_loss_box = tf.reduce_mean(tf.reduce_sum(rpn1_smooth_l1, reduction_indices=[1, 2, 3]))





        ##--------------------------------------------RPN-2-----------------------------------------------------------##
        ## classification loss
        #chris: use original score
        #rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])

        #chris: use added up score
        rpn_cls_score = tf.reshape(self.net.get_output('rpn12_cls_score_reshape'),[-1,2])

        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])

        #chris: reject here
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-2))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-2))),[-1])
        #chris: rejct done

        #chris: regular loss
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        #chris: done

        # #chris: second Focal Loss
        # length = tf.size(rpn_label)
        # rpn_label = tf.one_hot(indices = rpn_label, depth=2, on_value=1, off_value=0, axis=-1)
        # rpn_label = tf.cast(rpn_label, tf.float32)
        # rpn_weights = tf.ones([1, length], tf.float32)
        # rpn_cross_entropy = self.fl.compute_loss(prediction_tensor = rpn_cls_score, target_tensor = rpn_label, weights = rpn_weights)
        # #chris: Done


        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 



        ##----------------------------------------------R-CNN--------------------------------------------------------##
        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        # #final loss
        # #chris: original loss
        # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        # #chris

        # #chris
        # #new loss - rpn 1 classification
        # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn1_cross_entropy
        # #chris


        # chris
        # #new loss - rpn 1 classification + rpn 1 regression
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn1_cross_entropy + rpn1_loss_box
        #chris


        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)


        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            #chris I comment that
            # rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)
            #chris


            # #-------------------------------------------------chris: new BP CLS + REGRESSION---------------------------------------------------#
            rpn1_loss_cls_value, rpn1_loss_box_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn1_loss_box, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)
            #chris

            # #-------------------------------------------------chris: new BP CLS ONLY---------------------------------------------------#
            # rpn1_loss_cls_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)

            # #-------------------------------------------------------chris---------------------------------------------------#


            # #-------------------------------------------------Debug ONLY---------------------------------------------------#
            # a = sess.run([self.net.get_output('rpn_bbox_pred')],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)

            # #-------------------------------------------------------chris---------------------------------------------------#

            # print np.asarray(a).shape
            # print b
            # print c

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn1_loss_cls: %.4f, rpn1_loss_box: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value , rpn1_loss_cls_value, rpn1_loss_box_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

                # #chris
                # print 'rpn1_loss_cls: %.4f'%\
                #         (rpn1_loss_cls_value)
                # #chris



            #chris
            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        #chris
        print '\n'
        print '\n'
        print '\n'
        #chris

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'

        #train the model
        sw.train_model(sess, max_iters)

        print 'done solving'





