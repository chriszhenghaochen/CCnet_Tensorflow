#Cascade Chain Network

In this CC-Net, the cascaded classifier at a stage is aided by the
classification scores in previous stages. Feature chaining is further proposed so that the feature learning for the current
cascade stage uses the features in previous stages as the prior information. The chained ConvNet features and classifiers of
multiple stages are jointly learned in an end-to-end network.

## Requirement
* Tensorflow 1.0.0
* Titan X
* CUDA 8.0
* CUDNN 6

## Installation 
```
git clone https://github.com/chriszhenghaochen/CCnet_Tensorflow

cd path-to-your-folder/lib

make clean

make
```

## Run, Train and Test your model
Follow [***tf-faster-rcnn***](https://github.com/endernewton/tf-faster-rcnn) instruction to train your model

## Setting 
```
##########################CCNET SETTING########################
#------------FRCN Config--------#
__C.TRAIN.RPN_POST_NMS_TOP_N = 3000

#----------CCNet Config--------#
#REJECT
__C.RPN_REJECT3 = 9.9999999e-01
__C.RPN_REJECT2 = 9.99999e-01
__C.RPN_REJECT1 = 9.999e-01

__C.REJECT3 = 1.1 #not reject
__C.REJECT2 = 1.1 #not reject
__C.REJECT1 = 1.1 #not reject


#REJECT FACTOR
__C.RPN_REJECT3_FACTOR = 0.05
__C.RPN_REJECT2_FACTOR = 0.05
__C.RPN_REJECT1_FACTOR = 0.05

__C.REJECT3_FACTOR = 0.05
__C.REJECT2_FACTOR = 0.05
__C.REJECT1_FACTOR = 0.05


#CHAIN SCORE FACTOR
__C.SCORE_FACTOR1 = 0.2
__C.SCORE_FACTOR2 = 0.8


#TRAIN SETTING
__C.TRAIN.RPN_BATCH3 = 1024
__C.TRAIN.RPN_BATCH2 = 768
__C.TRAIN.RPN_BATCH1 = 512
__C.TRAIN.RPN_BATCH = 256

__C.TRAIN.BATCH3 = 1024
__C.TRAIN.BATCH2 = 768
__C.TRAIN.BATCH1 = 512
__C.TRAIN.BATCH = 256

__C.TRAIN.FOCAL_LOSS3 = False
__C.TRAIN.FOCAL_LOSS2 = False
__C.TRAIN.FOCAL_LOSS1 = False
__C.TRAIN.FOCAL_LOSS = False

__C.TRAIN.REPEAT = False

__C.TRAIN.OHEM3 = True
__C.TRAIN.OHEM2 = True
__C.TRAIN.OHEM1 = True
__C.TRAIN.OHEM = True

###############################################################
```

## Experiment Logs
Currently achieved ***76.8%*** for pascal_voc_2007 + pascal_voc_2012
I am runing newest experiment, logs will be released soon.

## Other Experments
I provide different architectures for you. in folder other experiments folder

## Resource
* Paper: [***Chained Cascade Network for Object Detection***](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ouyang_Chained_Cascade_Network_ICCV_2017_paper.pdf)

* [Official Caffe Version](https://github.com/wk910930/ccnn)

* Mxnet Version for Resnet CCnet (Project on going, will be released soon)

## Citation
```
@article{ouyang2017learning,
  title={Learning Chained Deep Features and Classifiers for Cascade in Object Detection},
  author={Ouyang, Wanli and Wang, Kun and Zhu, Xin and Wang, Xiaogang},
  journal={arXiv preprint arXiv:1702.07054},
  year={2017}
}
```

