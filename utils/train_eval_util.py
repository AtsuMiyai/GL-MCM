import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.dataset import Subset
import clip


def set_model_clip(args):
    model, _ = clip.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ImageNet'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'COCO_single':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_COCO_single'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'COCO_multi':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_COCO_multi'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == 'VOC_single':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(root, 'ID_VOC_single'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def get_subset_with_len(dataset, length, shuffle=False):
    dataset_size = len(dataset)
    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'IN22k':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'ImageNet-22K'), transform=preprocess)
    elif out_dataset == 'ood_voc':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'OOD_VOC'), transform=preprocess)
    elif out_dataset == 'ood_coco':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'OOD_COCO'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)    
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'Texture', 'images'),
                                        transform=preprocess)

    if args.num_ood_sumple > 0 and out_dataset != 'ood_voc':
        testsetout = get_subset_with_len(testsetout, length=args.num_ood_sumple, shuffle=True)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut
