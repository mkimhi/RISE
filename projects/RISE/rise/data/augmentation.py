import numpy as np
import logging
import sys
import random
from torchvision import transforms as tv_transform
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image
import torch
from torchvision import transforms as TVT

from detectron2.data import transforms as T
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

import copy

class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


def build_augmentation(cfg, is_train,strong=None):
    logger = logging.getLogger(__name__)
    aug_list = []
    if is_train:
        # Crop
        if cfg.INPUT.CROP.ENABLED:
            aug_list.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ms_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM if "by_clip" in cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING else 1
        aug_list.append(ResizeShortestEdge(min_size, max_size, sample_style, clip_frame_cnt=ms_clip_frame_cnt))

        # Flip
        if cfg.INPUT.RANDOM_FLIP != "none":
            if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
            else:
                flip_clip_frame_cnt = 1

            aug_list.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlip(
                    horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    clip_frame_cnt=flip_clip_frame_cnt,
                )
            )

        # Additional augmentations : brightness, contrast, saturation, rotation
        augmentations = cfg.INPUT.AUGMENTATIONS
        low,high = 0.9,1.1
        if strong:
            low,high = 0.5, 1.5
        
        if "brightness" in augmentations:
            aug_list.append(T.RandomBrightness(low,high))
        if "contrast" in augmentations:
            aug_list.append(T.RandomContrast(low,high))
        if "saturation" in augmentations:
            aug_list.append(T.RandomSaturation(low,high))
        if "rotation" in augmentations:
            aug_list.append(
                T.RandomRotation(
                    [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], sample_style="range"
                )
            )
        if strong:
            aug_list.append(T.RandomLighting(1.5))
        
        if not cfg.INPUT.CROP.ENABLED:
            return aug_list
        else:
            aug_no_crop = copy.deepcopy(aug_list)
            del aug_no_crop[0]
            return aug_no_crop, aug_list
    else:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        return aug_list

    
def create_strong_images(images):
        """
        Arguments:
            images <detectron2.structures.image_list.ImageList>: tensor of shape (2*batch_size ,3, h, w).
        Returns:
            images_s : a tensor of images after augmentations
            transform: the geometric transformation need to aply 
            on predictions of images for pseudo labeles
        """
        #define strong augmentations

        augs = TVT.Compose([
            TVT.RandomApply([TVT.ColorJitter([0.05, 0.95],[0.05, 0.95])],p=0.8),
            TVT.RandomApply([PlanckianJitter()],p=0.8),
            TVT.RandomGrayscale(p=0.2),
            TVT.RandomApply([TVT.GaussianBlur(5)],p=0.5),
        ])
        

        transformed_images = torch.stack([augs(img) for img in images.tensor])
        images_s = ImageList(transformed_images, image_sizes=images.image_sizes)
        return images_s



class PlanckianJitter:
    """Ramdomly jitter the image illuminant along the planckian locus"""

    def __init__(self, mode="blackbody", idx=None):
        self.idx = idx
        if mode == "blackbody":
            self.pl = np.array(
                [
                    [0.6743, 0.4029, 0.0013],
                    [0.6281, 0.4241, 0.1665],
                    [0.5919, 0.4372, 0.2513],
                    [0.5623, 0.4457, 0.3154],
                    [0.5376, 0.4515, 0.3672],
                    [0.5163, 0.4555, 0.4103],
                    [0.4979, 0.4584, 0.4468],
                    [0.4816, 0.4604, 0.4782],
                    [0.4672, 0.4619, 0.5053],
                    [0.4542, 0.4630, 0.5289],
                    [0.4426, 0.4638, 0.5497],
                    [0.4320, 0.4644, 0.5681],
                    [0.4223, 0.4648, 0.5844],
                    [0.4135, 0.4651, 0.5990],
                    [0.4054, 0.4653, 0.6121],
                    [0.3980, 0.4654, 0.6239],
                    [0.3911, 0.4655, 0.6346],
                    [0.3847, 0.4656, 0.6444],
                    [0.3787, 0.4656, 0.6532],
                    [0.3732, 0.4656, 0.6613],
                    [0.3680, 0.4655, 0.6688],
                    [0.3632, 0.4655, 0.6756],
                    [0.3586, 0.4655, 0.6820],
                    [0.3544, 0.4654, 0.6878],
                    [0.3503, 0.4653, 0.6933],
                ]
            )
        elif mode == "CIED":
            self.pl = np.array(
                [
                    [0.5829, 0.4421, 0.2288],
                    [0.5510, 0.4514, 0.2948],
                    [0.5246, 0.4576, 0.3488],
                    [0.5021, 0.4618, 0.3941],
                    [0.4826, 0.4646, 0.4325],
                    [0.4654, 0.4667, 0.4654],
                    [0.4502, 0.4681, 0.4938],
                    [0.4364, 0.4692, 0.5186],
                    [0.4240, 0.4700, 0.5403],
                    [0.4127, 0.4705, 0.5594],
                    [0.4023, 0.4709, 0.5763],
                    [0.3928, 0.4713, 0.5914],
                    [0.3839, 0.4715, 0.6049],
                    [0.3757, 0.4716, 0.6171],
                    [0.3681, 0.4717, 0.6281],
                    [0.3609, 0.4718, 0.6380],
                    [0.3543, 0.4719, 0.6472],
                    [0.3480, 0.4719, 0.6555],
                    [0.3421, 0.4719, 0.6631],
                    [0.3365, 0.4719, 0.6702],
                    [0.3313, 0.4719, 0.6766],
                    [0.3263, 0.4719, 0.6826],
                    [0.3217, 0.4719, 0.6882],
                ]
            )
        else:
            raise ValueError(
                'Mode "' + mode + '" not supported. Please choose between "blackbody" or "CIED".')

    def __call__(self, x):

        if isinstance(x, Image.Image):
            image = np.array(x.copy())
            image = image / 255
        else:
            image = x.clone().detach()

        if self.idx is not None:
            idx = self.idx
        else:
            idx = torch.randint(0, self.pl.shape[0], (1,)).item()
        # idx = np.random.randint(0, self.pl.shape[0])

        image[:, :, 0] = image[:, :, 0] * (self.pl[idx, 0] / self.pl[idx, 1])
        image[:, :, 2] = image[:, :, 2] * (self.pl[idx, 2] / self.pl[idx, 1])
        image[image > 1] = 1

        if isinstance(x, Image.Image):
            image = Image.fromarray(np.uint8(image * 255))

        return image

    def __repr__(self):
        if self.idx is not None:
            return (
                self.__class__.__name__
                + "( mode="
                + self.mode
                + ", illuminant="
                + np.array2string(self.pl[self.idx, :])
                + ")"
            )
        else:
            return self.__class__.__name__ + "(" + self.mode + ")"

    
