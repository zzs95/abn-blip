from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
# from lavis.datasets.datasets.capfilt_dataset import CapFiltCaptionInstructDataset, CapFiltCaptionDataset
from lavis.datasets.datasets.imgfeat_caption_datasets import (
    imgfeat_CapDataset,
    imgfeat_CapInstructDataset,
    imgfeat_CapEvalDataset,
    NoCapsEvalDataset,
)
from lavis.common.registry import registry

@registry.register_builder("imgfeat_caption")
class imgfaet_CapBuilder(BaseDatasetBuilder):
    train_dataset_cls = imgfeat_CapDataset
    eval_dataset_cls = imgfeat_CapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/imgfeat/defaults_cap.yaml",
    }