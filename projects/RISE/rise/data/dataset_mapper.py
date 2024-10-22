import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
##add cp to here!!
#from util.cut_paste import CopyPasteBank,image_obj,tv_save_image,save_imgs
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["YTVISDatasetMapper","ARMBENCHDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,

    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        if augmentations_nocrop is not None:
            self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
        else:
            self.augmentations_nocrop   = None
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train: #mk: check!!!!!!!!!!
            augs_nocrop, augs = build_augmentation(cfg, is_train)
        else:
            augs = build_augmentation(cfg, is_train)
            augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.RISE.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations. mk: add breakpoint here
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = selected_augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))



            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict
    

class YTVISDatasetMapper_unlabeled:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        augmentations_strong: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop_strong = None,
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        semi_supervised = True
        
    ):
        """
        Copy of the regular YTVISMAPPER but with two augmentations and no annotations
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        if augmentations_nocrop is not None:
            self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
        else:
            self.augmentations_nocrop   = None
        self.augmentations_strong          = T.AugmentationList(augmentations_strong)
        if augmentations_nocrop_strong is not None:
            self.augmentations_nocrop_strong   = T.AugmentationList(augmentations_nocrop_strong)
        else:
            self.augmentations_nocrop_strong   = None

        self.image_format           = image_format
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations} for weak and {augmentations_strong} for strong")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        if cfg.INPUT.CROP.ENABLED and is_train: #mk: check!!!!!!!!!!
            s_augs_nocrop, s_augs = build_augmentation(cfg, is_train,strong=True)
            w_augs_nocrop, w_augs = build_augmentation(cfg, is_train,strong=False)
        else:
            s_augs = build_augmentation(cfg, is_train,strong=True)
            w_augs = build_augmentation(cfg, is_train,strong=False)
            s_augs_nocrop,w_augs_nocrop = None, None

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": w_augs,
            "augmentations_nocrop": w_augs_nocrop,
            "augmentations_strong": s_augs,
            "augmentations_nocrop_strong": s_augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.RISE.NUM_CLASSES,
            "semi_supervised": True,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)


        #dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dataset_dict_s = copy.deepcopy(dataset_dict)
        file_names = dataset_dict.pop("file_names", None)
        file_names_s = dataset_dict_s.pop("file_names", None)
        dataset_dict["image"], dataset_dict_s["image"] = [], []
        dataset_dict["file_names"], dataset_dict_s["file_names"] = [], []
        
        #TODO: implement weak and strong augmentations strategies!!
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations

        if self.augmentations_nocrop_strong is not None and np.random.rand() > 0.5:
            selected_augmentations_strong = self.augmentations_nocrop_strong
        else:
            selected_augmentations_strong = self.augmentations_strong

        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            dataset_dict_s["file_names"].append(file_names[frame_idx])
            
            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            image_s = utils.read_image(file_names[frame_idx], format=self.image_format)

            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            aug_input_s = T.AugInput(image_s)

            transforms = selected_augmentations(aug_input)
            transforms_strong = selected_augmentations_strong(aug_input_s)
            #TODO: use transform strong for prediction transformation

            image = aug_input.image
            image_strong = aug_input_s.image

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
            dataset_dict_s["image"].append(torch.as_tensor(np.ascontiguousarray(image_strong.transpose(2, 0, 1))))

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            #dataset_dict_w["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
            #dataset_dict_s["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
        
        return dataset_dict #,dataset_dict_s
        return dataset_dict_w, dataset_dict_s #should not come here!!


class ARMBENCHDatasetMapper:
    """
    A callable which takes a dataset dict in ARMBENCH Dataset format,
    and map it into a format used by the detectron model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        num_classes: int = 2,
        
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        if augmentations_nocrop is not None:
            self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
        else:
            self.augmentations_nocrop   = None
        self.sequnce_augmentation = T.AugmentationList([T.RandomRotation([-3,3])]) #todo: check if works
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        #for sequnce modeling
        #self.cp_bank = CopyPasteBank()

        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train: #mk: check!!!!!!!!!!
            augs_nocrop, augs = build_augmentation(cfg, is_train)
        else:
            augs = build_augmentation(cfg, is_train)
            augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "num_classes": cfg.MODEL.RISE.NUM_CLASSES,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one frame

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations. mk: add breakpoint here
        #print("not deep copy?")
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            pass #todo: generate sequnce of frames from augmentations

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_names,file_names] #file_names
        
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations

            # Read image
        image = utils.read_image(file_names, format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = selected_augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
        #delete?
        dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))


        # # NOTE copy() is to prevent annotations getting changed from applying augmentations
        # _frame_annos = []
        # _anno = {}
        # for k, anno in enumerate(annos):
        #     _anno[k] = copy.deepcopy(anno)
        # _frame_annos.append(_anno)

        # USER: Implement additional transformations if you have other types of data
        augment_annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in annos
                if obj.get("iscrowd", 0) == 0]

        #print(augment_annos[0].keys())
        #print(f'annot len {len(augment_annos)}')
        ids = [augment_annos[id]['id'] for id in range(len(augment_annos))] 
        #print(ids, type(ids))
        instances = utils.annotations_to_instances(augment_annos, image_shape, mask_format="bitmask")
        #print(instances)
        instances.gt_ids = torch.tensor(ids)
        if instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = filter_empty_instances(instances)
        else:
            instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
        dataset_dict["instances"].append(instances)
        dataset_dict["instances"].append(instances)

        return dataset_dict
    
