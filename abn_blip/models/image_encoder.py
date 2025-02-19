import copy

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
import torchvision

from merlin.models import i3res


class ImageClassifier(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        resnet = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.i3_resnet = i3res.I3ResNet(
            copy.deepcopy(resnet), class_nb=out_channels, conv_class=True, return_pool=False # extract pooled feature
        )
        del resnet
        
    def forward(self, image):
        contrastive_features, ehr_features = self.i3_resnet(image)
        return contrastive_features, ehr_features