import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]
factor = 0.75

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2'))

             # #chris comment this
             # .conv(3, 3, 512, 1, 1, name='conv5_3'))
             # #chris

        #chris 
        #add new RPN here
        #========= RPN 1 ============
        (self.feed('conv5_2')
             .conv(3,3,512,1,1,name='rpn1_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn1_cls_score'))

        (self.feed('rpn1_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn1-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn1_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn1_bbox_pred'))

        #output postive and negative prob
        #========= RoI Proposal ============
        (self.feed('rpn1_cls_score')
             .reshape_layer(2,name = 'rpn1_cls_score_reshape')
             .softmax(name='rpn1_cls_prob'))


        # this is for Fast RCNN, not got RPN traininig
        # #chris
        # #reject layer
        # #chris 

        #chris: I leave that just in case, but now it is useless
        (self.feed('rpn1_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn1_cls_prob_reshape'))
        #chirs

        # (self.feed('rpn1_cls_prob_reshape','rpn1_bbox_pred','im_info')
        #      .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn1_rois'))

        # (self.feed('rpn1_rois','gt_boxes')
        #      .proposal_target_layer(n_classes,name = 'roi1-data'))
        # #chris


        #chris continue conv here 5_2 -> 5_3
        (self.feed('conv5_2')
            .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #chris
        

        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        #chris
        #cascade here
        (self.feed('rpn_cls_score','gt_boxes','im_info','data', 'rpn1_cls_prob_reshape')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data'))
        #chris

        # #chris
        # #do not apply Cascade in RPN
        # (self.feed('rpn_cls_score','gt_boxes','im_info','data')
        #      .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))
        # #chris

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))


        #chris: score add up
        (self.feed('rpn1_cls_score_reshape', 'rpn_cls_score_reshape')
             .scoreaddup(factor, name = 'rpn12_cls_score_reshape')
             .softmax(name='rpn12_cls_prob'))

        (self.feed('rpn12_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn12_cls_prob_reshape'))
        #chris



        # (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info','rpn1_cls_prob_reshape')
        #      .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))


        #chris: proposal add up
        (self.feed('rpn12_cls_prob_reshape','rpn_bbox_pred','im_info','rpn1_cls_prob_reshape')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))
        #chris

        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))