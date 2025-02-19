"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from collections import OrderedDict
from lavis.datasets.datasets.base_dataset import BaseDataset
import numpy as np
from PE_datasets.normal_text_template import normal_text_list
from abnormality_list import abnormality_list

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class imgfeat_CapDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["AccessionNumber_md5"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.normal_text_list = normal_text_list
        # self.normal_text = self.text_processor('No findings.') # TODO normal text arugmentation
        
    def __getitem__(self, index):
        ann = self.annotation[index]
        # img_feat_ = ann["image_feat"][1:].replace('_train', '_train_noA')
        image_feat_path = os.path.join(self.vis_root, ann["image_feat"][1:])
        try:
            image_feat = np.load(image_feat_path)
        except:
            return None # image does not exist
        
        # image = Image.fromarray((np.random.rand(640,500,3)*255).astype(np.uint8))
        # image = self.vis_processor(image)
        abn_label = np.array(ann['abn_label'])
        # caption_list = [self.normal_text]*len(ann['abn_label'])
        # for i, abn_i in enumerate(np.argwhere(abn_label).reshape(-1)):
        #     caption_list[abn_i] = self.text_processor(ann["abn_text"][i])
        
        caption_list = np.random.choice(self.normal_text_list, len(abnormality_list)).tolist()
        for i, abn_name in enumerate(abnormality_list):
            caption_list[i] = caption_list[i].replace('<ABN_FIND>', abn_name)

        for i, abn_i in enumerate(np.argwhere(abn_label).reshape(-1)):
            caption_list[abn_i] = self.text_processor(ann["abn_text"][i])
        
        return {
            # "image": image,
            "feat1": torch.from_numpy(image_feat['pooled_scale_1']),
            "feat2": torch.from_numpy(image_feat['pooled_scale_2']),
            "feat3": torch.from_numpy(image_feat['pooled_scale_3']),
            "feat4": torch.from_numpy(image_feat['pooled_scale_4']),
            "feat5": torch.from_numpy(image_feat['scale_5']),
            "abn_text_input": caption_list,
            "abn_label": torch.from_numpy(abn_label),
            "image_id": self.img_ids[ann["AccessionNumber_md5"]],
        }

class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

class imgfeat_CapInstructDataset(imgfeat_CapDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data


class imgfeat_CapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["AccessionNumber_md5"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_feat_path = os.path.join(self.vis_root, ann["image_feat"][1:])
        try:
            image_feat = np.load(image_feat_path)
        except:
            return None # image does not exist
        
        abn_label = np.array(ann['abn_label'])
        
        caption_list = ['no findings of <ABN_FIND>.'] * len(abnormality_list)
        for i, abn_name in enumerate(abnormality_list):
            caption_list[i] = caption_list[i].replace('<ABN_FIND>', abn_name)

        for i, abn_i in enumerate(np.argwhere(abn_label).reshape(-1)):
            caption_list[abn_i] = self.text_processor(ann["abn_text"][i])
        
        return {
            "feat1": torch.from_numpy(image_feat['pooled_scale_1'].astype(np.float32)),
            "feat2": torch.from_numpy(image_feat['pooled_scale_2'].astype(np.float32)),
            "feat3": torch.from_numpy(image_feat['pooled_scale_3'].astype(np.float32)),
            "feat4": torch.from_numpy(image_feat['pooled_scale_4'].astype(np.float32)),
            "feat5": torch.from_numpy(image_feat['scale_5'].astype(np.float32)),
            "abn_text_input": caption_list,
            "abn_label": torch.from_numpy(abn_label),
            "image_id": ann["AccessionNumber_md5"],
            "image_path": ann["image_feat"],
        }




class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }