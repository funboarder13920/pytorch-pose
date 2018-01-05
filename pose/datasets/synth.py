from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *


class Synth(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, sigma=1,
                 zoom_factor=0.2, rot_factor=30, shift_factor=[30, 30], label_type='Gaussian'):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.zoom_factor = zoom_factor
        self.shift_factor = shift_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './data/synth/mean_segm.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        zf = self.zoom_factor
        rf = self.rot_factor
        sxf, syf = self.shift_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # For single-person pose estimation with a centered/scaled figure
        img = load_image(img_path)  # CxHxW

        r = 0
        zoom = 1
        shift = [0, 0]
        if self.is_train:
            zoom = torch.randn(1).mul_(zf).add(1).clamp(0.6, 1.5)[0] if random.random() <= 0.6 else 1
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            if random.random() <= 0.6:
              sx = torch.randn(1).mul_(syf).clamp(-2*syf, 2*syf)[0]
              sy = torch.randn(1).mul_(sxf).clamp(-2*sxf, 2*sxf)[0]
              shift = [int(sx), int(sy)]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = augment_data(img,[self.inp_res, self.inp_res], zoom=zoom, rot=r, shift=shift)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        img_target_path = os.path.join(self.img_folder, a['img_target_paths'].replace('.png', '.exr'))
        img_target = load_exr(img_target_path)
        img_target = augment_data(img_target, [self.out_res, self.out_res], zoom=zoom, rot=r, shift=shift)

        target = to_grey(img_target)

        # Meta info
        meta = {'index' : index}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
