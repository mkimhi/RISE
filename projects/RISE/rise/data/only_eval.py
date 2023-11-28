import os, json
import numpy as np
import matplotlib.pyplot as plt

from datasets.armbench import ArmBench

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


input_dir = 'datasets/armbench'
coco_gt = ArmBench(os.path.join(input_dir,'test_partial.json'))
res = os.path.join(input_dir,'results.json')
coco_dt = coco_gt.loadRes(res)

cocoEval = COCOeval(coco_gt,coco_dt,'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()