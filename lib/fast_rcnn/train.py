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

pass_threshold = 0.3
printRecall = False

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


    def train_model_recall(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        #chris
        # RPN-1
        # classification loss
        rpn1_cls_score = tf.reshape(self.net.get_output('rpn1_cls_score_reshape'),[-1,2])

        # chris
        # rpn1_label = tf.reshape(self.net.get_output('rpn1-data')[0],[-1])
        rejectlabel = tf.reshape(self.net.get_output('rpn1-data')[0],[-1])
        rpn1_label =  rejectlabel
        #chris

        rpn1_cls_prob = tf.reshape(self.net.get_output('rpn1_cls_prob_reshape'),[-1,2])[:,1]

        #chris
        #rpn reject step-1 negative sample, postive prob > 0.3
        rpn1_pass_inds = self.passSample(rpn1_cls_prob, threshold = pass_threshold)
        rpn1_neg_inds = self.rejectSample(rpn1_cls_prob, threshold = pass_threshold)
        #chris


        #reject step-2 0.3 <IOU <0.7
        rpn1_cls_score = tf.reshape(tf.gather(rpn1_cls_score,tf.where(tf.not_equal(rpn1_label,-1))),[-1,2])
        rpn1_label = tf.reshape(tf.gather(rpn1_label,tf.where(tf.not_equal(rpn1_label,-1))),[-1])
        rpn1_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn1_cls_score, labels=rpn1_label))



        # # bounding box regression L1 loss
        # rpn1_bbox_pred = self.net.get_output('rpn1_bbox_pred')
        # rpn1_bbox_targets = tf.transpose(self.net.get_output('rpn1-data')[1],[0,2,3,1])
        # rpn1_bbox_inside_weights = tf.transpose(self.net.get_output('rpn1-data')[2],[0,2,3,1])
        # rpn1_bbox_outside_weights = tf.transpose(self.net.get_output('rpn1-data')[3],[0,2,3,1])

        # rpn1_smooth_l1 = self._modified_smooth_l1(3.0, rpn1_bbox_pred, rpn1_bbox_targets, rpn1_bbox_inside_weights, rpn1_bbox_outside_weights)
        # rpn1_loss_box = tf.reduce_mean(tf.reduce_sum(rpn1_smooth_l1, reduction_indices=[1, 2, 3]))
        # #chris




        # RPN
        # classification loss
        rpn_cls_score1 = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label1 = tf.reshape(self.net.get_output('rpn-data')[0],[-1])

        #chris 
        #reject here:
        rpn_cls_score2 = tf.reshape(tf.gather(rpn_cls_score1, tf.reshape(rpn1_pass_inds,[-1])),[-1,2])
        rpn_label2 = tf.reshape(tf.gather(rpn_label1, tf.reshape(rpn1_pass_inds,[-1])),[-1])
        #chris


        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score2,tf.where(tf.not_equal(rpn_label2,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label2,tf.where(tf.not_equal(rpn_label2,-1))),[-1])

        #chris 
        # reject will happen here:
        # sub_ind = self.reject(rpn1_cls_score)
        # rpn_label = rpn1_label[sub_ind]
        # rpn_cls_score = rpn1_label[sub_ind]
        # chris

        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 


        # R-CNN
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

        # final loss
        #chris I comment that
        #loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        #chris

        # #chris
        # #this is the new loss
        # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn1_cross_entropy + rpn1_loss_box
        # #chris

        #chris
        #this is the new loss with rpn1-cls only
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn1_cross_entropy
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

        #chris
        #for ploting
        recalls = []
        rejects = []
        #chris

        #chris 
        #for testing
        for iter in range(max_iters):
        # for iter in range(1000):
        #chris
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


            # #chris 
            # #----------------------------------------------chris: new BP CLS + REG--------------------------------------------------#
            # rpn1_loss_cls_value, rpn1_loss_box_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn1_loss_box, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)
            # #-------------------------------------------------chris--------------------------------------------------------#



            #chris reject


            #chris


            # #-------------------------------------------------chris: new BP CLS ONLY---------------------------------------------------#
            # rpn1_loss_cls_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)

            # #-------------------------------------------------------chris---------------------------------------------------#

            # #chris -- Debug
           
            # a,b,c,d,e = sess.run([rpn1_cls_score, rpn1_label, self.net.get_output('rpn_cls_prob'), self.net.get_output('rpn_cls_prob_reshape'), self.reject(rpn1_cls_score)],
            #                                                             feed_dict=feed_dict,
            #                                                             options=run_options,
            #                                                             run_metadata=run_metadata)                    
            # print 'rpn score'
            # print a
            # print a.shape
            # print 'rpn label'
            # print b
            # print b.shape
            # # print 'softmax result'
            # # print c
            # # print c.shape
            # print 'after reject'
            # print e.shape
            # print e
            # #chris


            # #chris -- Debug
           
            # a,b,c,d,e = sess.run([rpn_cls_score1,rpn_cls_score2,rpn_cls_score, rpn1_pass_inds, rpn1_cls_prob],
            #                                                             feed_dict=feed_dict,
            #                                                             options=run_options,
            #                                                          run_metadata=run_metadata)                    
            # # print 'rpn score 1'
            # # # print a
            # # print a.shape
            # # print 'rpn score 2'
            # # # print b
            # # print b.shape
            # # print 'rpn score 3'
            # # # print c
            # # print c.shape
            # # print 'pass index'
            # # print d
            # #print d.shape

            # print 'prob'
            # print e
            # #chris

            #--------------------------------------for recall--------------------------------#
            #chris -- print reject sample number
            # a,b = sess.run([self.recall1(rpn1_neg_inds, rejectlabel),
            #                 self.recall2(rpn1_neg_inds, rejectlabel)],
            #                 feed_dict=feed_dict,
            #                 options=run_options,
            #                 run_metadata=run_metadata)  

            a,b, rpn1_loss_cls_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([self.recall1(rpn1_neg_inds, rejectlabel),
                                                                                                                           self.recall2(rpn1_neg_inds, rejectlabel),
                                                                                                                           rpn1_cross_entropy, 
                                                                                                                           rpn_cross_entropy, 
                                                                                                                           rpn_loss_box, 
                                                                                                                           cross_entropy, 
                                                                                                                           loss_box, train_op],
                                                                                                                feed_dict=feed_dict,
                                                                                                                options=run_options,
                                                                                                                run_metadata=run_metadata)


            num_reject = a.shape[0]

            print 'I am sure this is changed '
            print 'RPN conv5_2 -> RPN con5_3, Reject %d Neative Samples'%\
            (num_reject)


            # print 'Recall of Correct Reject is %d'%\
            # print 'c'
            # print c.shape
            # print 'd'
            # print d.shape

            recall = float(b.shape[0])/a.shape[0]
            # print 'Reject Sample Recall is %.4f'%\
            # (recall)

            recalls.append(recall)
            rejects.append(num_reject)

            # print '\n'
            # print '\n'

            #chris


            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

                # #chris
                # print 'rpn1_loss_cls: %.4f, rpn1_loss_box: %.4f'%\
                #         (rpn1_loss_cls_value , rpn1_loss_box_value)
                # #chris

                #chris, rpn1_cls only
                print 'rpn1_loss_cls: %.4f'%\
                        (rpn1_loss_cls_value)

                # print '\n'
                # print '\n'
                #chris



            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        plt.plot(recalls)
        plt.show()
        plt.plot(rejects)
        plt.show()
        

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        #chris
        # RPN-1
        # classification loss
        rpn1_cls_score = tf.reshape(self.net.get_output('rpn1_cls_score_reshape'),[-1,2])
        rpn1_label = tf.reshape(self.net.get_output('rpn1-data')[0],[-1])
        rpn1_cls_score = tf.reshape(tf.gather(rpn1_cls_score,tf.where(tf.not_equal(rpn1_label,-1))),[-1,2])
        rpn1_label = tf.reshape(tf.gather(rpn1_label,tf.where(tf.not_equal(rpn1_label,-1))),[-1])
        rpn1_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn1_cls_score, labels=rpn1_label))

        # chris: 
        #I don't need that
        # # bounding box regression L1 loss
        # rpn1_bbox_pred = self.net.get_output('rpn1_bbox_pred')
        # rpn1_bbox_targets = tf.transpose(self.net.get_output('rpn1-data')[1],[0,2,3,1])
        # rpn1_bbox_inside_weights = tf.transpose(self.net.get_output('rpn1-data')[2],[0,2,3,1])
        # rpn1_bbox_outside_weights = tf.transpose(self.net.get_output('rpn1-data')[3],[0,2,3,1])

        # rpn1_smooth_l1 = self._modified_smooth_l1(3.0, rpn1_bbox_pred, rpn1_bbox_targets, rpn1_bbox_inside_weights, rpn1_bbox_outside_weights)
        # rpn1_loss_box = tf.reduce_mean(tf.reduce_sum(rpn1_smooth_l1, reduction_indices=[1, 2, 3]))
        # #chris
        # chris



        ## RPN
        ## classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))


        # #----------------------chris RPN Reject-----------------------#
        # rpn_label_reject = tf.reshape(self.net.get_output('rpn-data-reject')[0],[-1])
        # rpn_cls_score_reject = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label_reject,-1))),[-1,2])
        # rpn_label_reject = tf.reshape(tf.gather(rpn_label_reject,tf.where(tf.not_equal(rpn_label_reject,-1))),[-1])
        # rpn_cross_entropy_reject = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_reject, labels=rpn_label_reject))



        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 
        # R-CNN
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

        # final loss
        #chris I comment that
        #loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        #chris

        #chris
        #this is the new loss
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + rpn1_cross_entropy
        #chris


        # #chris
        # #this is for reject loss
        # loss_reject = cross_entropy + loss_box + rpn_cross_entropy_reject + rpn_loss_box + rpn1_cross_entropy
        # #chris


        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)

        # #chris
        # #reject train op:
        # train_op_reject = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_reject, global_step=global_step)
        # #chris

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


            # #chris 
            # #new BP
            # rpn1_loss_cls_value, rpn1_loss_box_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn1_loss_box, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
            #                                                                                     feed_dict=feed_dict,
            #                                                                                     options=run_options,
            #                                                                                     run_metadata=run_metadata)
            # #chris

            #-------------------------------------------------chris: new BP CLS ONLY---------------------------------------------------#
            rpn1_loss_cls_value, rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn1_cross_entropy, rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)

            #-------------------------------------------------------chris---------------------------------------------------#


            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

                #chris
                print 'rpn1_loss_cls: %.4f'%\
                        (rpn1_loss_cls_value)
                #chris



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

        #chris
        #print recall:
        if printRecall:
            sw.train_model_recall(sess, max_iters)

        #train
        else:
            sw.train_model(sess, max_iters)

        print 'done solving'





