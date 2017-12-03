# CCnet
Replace lib folder in the [***Faster-RCNN_TF***](https://github.com/smallcorgi/Faster-RCNN_TF) and follow in instruction to run

# Tensorflow
now support Tensorflow 1.4.0

# Paper
Ouyang, W., Wang, K., Zhu, X., Wang, X. (2017). Chained Cascade Network for Object Detection. International Conference on Computer Vision (ICCV 2017), Italy.

https://arxiv.org/abs/1702.07054


# Result
Enhance baseline (train: VOC_2007, test: VOC_2007, VGG16) from ***68.1%*** to ***71.5%*** (In paper is 72.4%)

# Fine-Tuning
In Config file

```
###### CCNet Config #######
#Enable OHEM
__C.TRAIN.RPN_OHEM = True

#train setting
__C.TRAIN.boxChain = False
__C.TRAIN.FACTOR = 0
__C.TRAIN.REJECT = 0.3

#test setting
__C.TEST.FACTOR = 0
__C.TEST.REJECT = 0.3
__C.TEST.boxChain = False

#loss
__C.TRAIN.LOSS = 'Focal Loss'
############################
```
