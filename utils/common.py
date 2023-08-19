import torch
import os
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_test_labels(args):
    if args.in_dataset == 'ImageNet':
        test_labels = obtain_ImageNet_classes()
    elif args.in_dataset == 'COCO_single':
        test_labels = obtain_COCO_single_classes()
    elif args.in_dataset == 'COCO_multi':
        test_labels = obtain_COCO_multi_classes()
    elif args.in_dataset == 'VOC_single':
        test_labels = obtain_VOC_single_classes()
    return test_labels


def obtain_ImageNet_classes():
    loc = os.path.join('data', 'ImageNet')
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls


def obtain_COCO_single_classes():
    class_name = [
       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return class_name


def obtain_COCO_multi_classes():
    class_name = [
       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return class_name


def obtain_VOC_single_classes():
    class_name = [
         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    return class_name


def accuracy(output, target, topk=(1,)):
    '''Computes the precision@k for the specified values of k'''
    maxk = max(topk)
    batch_size = target.size(0)
    # values, indices = input.topk(k, dim=1, largest=True, sorted=True)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def read_file(file_path, root='corpus'):
    corpus = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            corpus.append(line[:-1])
    return corpus


def calculate_cosine_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
