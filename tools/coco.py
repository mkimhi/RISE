import os
from dataclasses import dataclass

import imageio
import numpy as np
from torch.utils.data import Dataset

coco_available = False
try:
    from pycocotools.coco import COCO

    coco_available = True
except ModuleNotFoundError:
    pass



@dataclass
class CocoDatasetConfig:
    """
    The config class for COCO.
    """

    _target_: str = "aps_models.data.datasets.coco.CocoDataset"
    base_path: str = ""
    convert_to_tami_convention: bool = True




class CocoDataset(Dataset):
    """Class to handle loading of data from COCO dataset"""

    def __init__(self, **kwargs):
        """ """

        super().__init__()
        if not coco_available:
            raise RuntimeError("PyCocoTools is not install. Run 'poetry install -E coco'")

        # extract base path
        self.base_path = kwargs["base_path"]
        self.convert_to_tami_convention = kwargs["convert_to_tami_convention"]
        # data paths
        self.images_path = os.path.join(self.base_path, "images")
        #self.annotations_path = os.path.join(self.base_path, "annotations", "instances.json")
        self.annotations_path = os.path.join(self.base_path, "train.json")

        # load data via provided api
        self.cocoapi = COCO(self.annotations_path)

        self.label_map = {}
        for idx, cat in enumerate(self.cocoapi.dataset["categories"], 1):
            self.label_map[cat["id"]] = idx

        self.imgs_keys = []
        for idx in self.cocoapi.imgs:
            self.imgs_keys.append(idx)

        # dataset info
        self.imgs_keys = sorted(self.imgs_keys)
        self.dset_length = len(self.imgs_keys)

    def __getitem__(self, index):
        """
        Load data, extract annotated information and map them to TAMI format
        """
        # extract coco internal index
        coco_index = self.imgs_keys[index]

        coco_image = self.cocoapi.loadImgs(coco_index)
        coco_annotation = self.cocoapi.imgToAnns[coco_index]
        coco_categories = self.cocoapi.cats

        # load image
        fname_image = coco_image[0]["file_name"]
        fpath_image = os.path.join(self.images_path, fname_image)
        rgb = imageio.v2.imread(fpath_image).astype(np.float32)
        sample = {"images": {"rgb": rgb}}

        if self.convert_to_tami_convention:

            sample["object_class_names"] = []
            sample["object_class_ids"] = []
            sample["object_ids"] = []
            sample["object_names"] = []
            sample["boxes"] = []
            sample["masks"] = []
            sample["iscrowd"] = []

            out_annotation = []
            # process annotations corresponding to indexed image
            for annotation in coco_annotation:
                sample["object_class_names"].append(coco_categories[annotation["category_id"]]["name"])
                sample["object_class_ids"].append(self.label_map[annotation["category_id"]])
                sample["object_ids"].append(annotation["id"])
                sample["iscrowd"].append(annotation["iscrowd"])
                bbox = annotation["bbox"]  # box ltwh
                sample["boxes"].append(
                    np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
                )  # ltrb
                sample["masks"].append(self.cocoapi.annToMask(annotation)[:, :, np.newaxis])  # already uint8 [0, 1]

        else:
            out_annotation = {"images": coco_image, "annotations": coco_annotation, "categories": [coco_categories]}
            sample["coco_annotation"] = out_annotation

        sample["image_id"] = index

        return sample

    def __len__(self):
        return self.dset_length
