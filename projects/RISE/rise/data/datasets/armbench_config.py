# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
import random
from .armbench import ArmBench

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse ARMBENCH dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_armbench_json", "register_armbench_instances"]



ARMBENCH_CATEGORIES = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "tote"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "object"},
]



def _get_armbench_instances_meta():
    thing_ids = [k["id"] for k in ARMBENCH_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ARMBENCH_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ARMBENCH_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def load_armbench_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None,old_load=True):
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        armbench_api = ArmBench(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(armbench_api.getCatIds())
        cats = armbench_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])] # type: ignore
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    _ids = sorted(armbench_api.imgs.keys())

    imgs = armbench_api.loadImgs(_ids)

    anns = [armbench_api.imgToAnns[id] for id in _ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(armbench_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in ARMBENCH format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (im_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_names"] = os.path.join(image_root, im_dict["file_name"])
        record["height"] = im_dict["height"]
        record["width"] = im_dict["width"]
        id = record["id"] = im_dict["id"]
        #TODO: change for SSL
        """if 'unlabeled' not in dataset_name and 'val' not in dataset_name and 'test' not in dataset_name and not old_load:
            continue"""
        frame_objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == id

            obj = {key: anno[key] for key in ann_keys if key in anno}

            bbox = anno.get("bbox", None)
            segm = anno.get("segmentation", None)

            if not (bbox and segm):
                continue

            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYWH_ABS

            if isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    segm = mask_util.frPyObjects(segm, *segm["size"])
            elif segm:
                # filter out invalid polygons (< 3 points)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    num_instances_without_valid_segmentation += 1
                    continue  # ignore this instance
            obj["segmentation"] = segm

            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            frame_objs.append(obj)

        record["annotations"] = frame_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_armbench_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in ARMBENCH's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "armbench_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_armbench_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="armbench", **metadata
    )


if __name__ == "__main__":
    """
    Test the ARMBENCH json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image
    save_annoteted = False
    fraction = 0.55
    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()