# CCnet
Replace lib folder in the [***tf-fast-rcnn***](https://github.com/endernewton/tf-faster-rcnn) and follow in instruction to run

# Tensorflow
now support Tensorflow 1.4.0

# Paper
Ouyang, W., Wang, K., Zhu, X., Wang, X. (2017). Chained Cascade Network for Object Detection. International Conference on Computer Vision (ICCV 2017), Italy.

https://arxiv.org/abs/1702.07054


# Result
Enhance baseline (train: VOC_2007, test: VOC_2007, VGG16) from ***69.7%*** to ***71.6%*** (In paper is 72.4%)

Enhance baseline (train: VOC_2007, VOC_2012 test: VOC_2007, VGG16) from ***75.3%*** to ***76.1%*** (In paper is 80.4%)

# Fine-Tuning
In Config file

```
#----------CCNet Config--------#
#train
__C.TRAIN.REJECT4_3 = 0.1
__C.TRAIN.REJECT5_2 = 0.2
__C.TRAIN.REJECT5_3 = 0.1

#test
__C.TEST.REJECT = 0

#score factor
__C.SCORE_FACTOR = 1

#boxTrain
__C.BOX_CHAIN = True

#ohem 3: 1 restrinct
__C.TRAIN.OHEM4_2 = False
__C.TRAIN.OHEM4_3 = False
__C.TRAIN.OHEM5_2 = False
__C.TRAIN.OHEM5_3 = True

#focal loss
__C.TRAIN.FOCAL_LOSS4_2 = False
__C.TRAIN.FOCAL_LOSS4_3 = False
__C.TRAIN.FOCAL_LOSS5_2 = False
__C.TRAIN.FOCAL_LOSS5_3 = False
#--------------done------------#
```
# Previous Version
If you want previous version (mAP,70.5%), please contact zhenghao.chen@sydney.edu.au directly.

# TODO:
* [***FPN***](https://github.com/xmyqsh/FPN) baseline 
* [***Refindet***](https://github.com/sfzhang15/RefineDet) Baseline 
* Fix ***Tensorboard*** Bug
* Add Proposal NMS, from [***SSD***](https://github.com/balancap/SSD-Tensorflow)

# Experiment Log
If you wanna know my current experiment log, please contact zhenghao.chen@sydney.edu.au directly.


