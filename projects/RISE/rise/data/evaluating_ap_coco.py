import os
import sys
from typing import Any, Dict

import numpy as np
import pycocotools.mask as mask_util
from aps_core.utils.logger import get_logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = get_logger()


def to_coco_annotation(annotation_id: int, annotation_type: str, **kw: Any) -> Dict[str, Any]:
    """Convert data into coco annotation format depending on the annoation type

    Args:
        annotation_id(int): annotation id. Must be positive.
        annotation_type(str): type of annotation ['image', 'detection', 'groundtruth']

    Depending on the type of annotation the following MUST be set
    Kw Args:
        image:
            height(int)
            width(int)

        detection:
            image_id(int): corresponding image id
            category_id(int): object category
            bbox([t, l, r, b]) or segmentation(array): box or segmentation prediction (the latter as RLE)
            score(float): prediction score

        groundtruth:
            image_id(int): corresponding image id
            category_id(int): object category
            bbox([t, l, r, b]) and/or segmentation(array): box or segmentation annotation (the latter as polygon or RLE)
            area(float): box area (if none given, it is inferred from the box)
            iscrowd(bit): 0 if segmentation is a polygon, 1 if is RLE

    Returns:
        Dict[str, Any]: annotation in coco format

    Raises:
        KeyError if correct kw args are not met
        ValueError if given annotation type not in ['image', 'detection', 'groundtruth']
        AssertionError if annotation id <= 0
    """
    assert annotation_id > 0, f'Invalid annotation id "{annotation_id}". Annotation id must be positive.'

    annotation: Dict[str, Any] = {"id": annotation_id}
    if annotation_type not in ["image", "detection", "groundtruth"]:
        raise ValueError(f'Given annotation type "{annotation_type}" not supported')

    if annotation_type == "image":
        annotation["height"] = kw["height"]
        annotation["width"] = kw["width"]
    else:
        annotation["image_id"] = kw["image_id"]
        annotation["category_id"] = kw["category_id"]
        # depending on the type either grab score or iscrowd

        if annotation_type == "detection":
            annotation["score"] = kw["score"]
        else:
            annotation["iscrowd"] = kw["iscrowd"]

        # check area (if none given try to infer from box)
        annotation["area"] = kw.get("area", None)
        # check for box
        box = kw.get("bbox", None)
        if box is not None:
            box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            annotation["bbox"] = box_coco
            if annotation["area"] is None:
                annotation["area"] = box_coco[2] * box_coco[3]
        # check segmentation
        segm = kw.get("segmentation", None)
        if segm is not None:
            rle = mask_util.encode(np.array(segm[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            annotation["segmentation"] = rle
        if segm is None and box is None:
            raise ValueError("Neither box nor segmentation info given")
    return annotation


class HiddenPrints:
    """
    A helper class to suppress outputs on stdout during coco evaluation
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CocoEvaluation:
    """
    A class to manage coco evaluations during training or validation runs.
    Supports adding batch results and evaluation after the whole epoch.
    """

    def __init__(self, categories: dict, mask_threshold: float = 0.5):
        """
        Args:
            mask_threshold(float): as coco expects binary masks, everything above mask_threshold is considered as 1
            categories(dict): object names in shape {0: "object_name1", 1: "object_name2", ...}
        """

        self.mask_threshold = mask_threshold
        self._coco_images: list = []
        self._target_anns: list = []
        self._pred_anns: list = []
        self._categories: list = []
        self._coco_eval: COCOeval = None
        for key, value in categories.items():
            self._categories.append({"id": key, "name": value})

    def add_batch(self, images: list, predictions: list, targets: list):
        """
        Adding batches of input images, predictions and targets

        Args:
            images(list): batch containing dicts with keys (id, width, height)
            predictions(list): batch containing dicts with keys (boxes, labels, scores, masks)
            targets(list): batch containing dicts with keys (bboxes, labels, masks)
        """

        for image, prediction, target in zip(images, predictions, targets):

            # create unique image id
            image_id = len(self._coco_images) + 1

            image_annotation = to_coco_annotation(
                annotation_id=image_id, annotation_type="image", width=image["width"], height=image["height"]
            )
            self._coco_images.append(image_annotation)

            pred_boxes = prediction["boxes"]
            pred_labels = prediction["labels"]
            pred_scores = prediction["scores"]

            num_objects = len(pred_boxes)

            if "masks" in prediction:
                pred_segmentation = [(mask > self.mask_threshold).astype(np.uint8) for mask in prediction["masks"]]
            else:
                pred_segmentation = [None] * num_objects

            assert len(pred_labels) == num_objects
            assert len(pred_scores) == num_objects

            for box, label, score, segmentation in zip(pred_boxes, pred_labels, pred_scores, pred_segmentation):
                prediction_annotation = to_coco_annotation(
                    annotation_id=len(self._pred_anns) + 1,
                    annotation_type="detection",
                    image_id=image_id,
                    category_id=label,
                    bbox=box,
                    score=score,
                    segmentation=segmentation,
                )

                self._pred_anns.append(prediction_annotation)

            target_boxes = target["boxes"]
            if len(target["boxes"]):
                num_objects = len(target_boxes)
                target_labels = target["labels"]

                if "masks" in target:
                    target_segmentation = target["masks"]
                else:
                    target_segmentation = [None] * num_objects

                assert len(target_labels) == num_objects

                for box, label, segmentation in zip(target_boxes, target_labels, target_segmentation):
                    gt_annotation = to_coco_annotation(
                        annotation_id=len(self._target_anns) + 1,
                        annotation_type="groundtruth",
                        image_id=image_id,
                        category_id=label,
                        bbox=box,
                        iscrowd=0,
                        segmentation=segmentation,
                    )
                    self._target_anns.append(gt_annotation)

    def eval(self, iouType: str = "bbox"):
        """
        Adding batches of input images, predictions and targets

        Args:
            iouType(str): coco evaluation type, "bbox" or "segm"
        """

        coco_gts = COCO()
        coco_gts.dataset["images"] = self._coco_images
        coco_gts.dataset["annotations"] = self._target_anns
        coco_gts.dataset["categories"] = self._categories
        coco_gts.createIndex()

        coco_preds = COCO()
        coco_preds.dataset["images"] = self._coco_images
        coco_preds.dataset["annotations"] = self._pred_anns
        coco_preds.dataset["categories"] = self._categories
        coco_preds.createIndex()

        # compute coco scores
        with HiddenPrints():
            E = COCOeval(coco_gts, coco_preds, iouType=iouType)
            E.evaluate()
            E.accumulate()
            E.summarize()
            ap = E.stats[0]
            ar = E.stats[8]

        self._coco_eval = E
        return ap, ar


def add_coco_batch(coco_eval, images, predictions, targets):
    image_info = []
    prediction_info = []
    target_info = []

    # for image, prediction, target in batch
    for image, prediction, target in zip(images, predictions, targets):
        image_info.append(
            {
                "id": target["image_id"],
                "width": image.size()[2],
                "height": image.size()[1],
            }
        )

        prediction_info.append(
            {
                "labels": prediction["labels"].detach().cpu().numpy(),
                "boxes": prediction["boxes"].detach().cpu().numpy(),
                "scores": prediction["scores"].detach().cpu().numpy(),
                "masks": prediction["masks"].detach().cpu().numpy(),  # .squeeze(),
            }
        )

        target_info.append(
            {
                "labels": target["labels"].detach().cpu().numpy(),
                "boxes": target["boxes"].detach().cpu().numpy(),
                "masks": target["masks"].detach().cpu().numpy(),
            }
        )

    coco_eval.add_batch(image_info, prediction_info, target_info)

categories = {1: "Tote", 2: "Object"}
coco_eval = CocoEvaluation(categories)
ap,ar = coco_eval.eval()
print(f'AP: {ap},AR: {ar}')