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
from random import randrange
import sys
from PIL import Image
from ytvis import _get_ytvis_2021_instances_meta,load_ytvis_json,register_ytvis_instances

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.data.datasets  # noqa # add pre-defined metadata
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from tools.coco import CocoDataset

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


def _list_to_dict(dicts):
    changed_frames_d = {}
    for i,d in enumerate(dicts):
        vid_name = d["file_names"][0].split('/')[-2]
        vid_id = d['video_id'] #check by id
        prev_len = len(d['annotations'][0])
        found=False 
        for j,id in enumerate(d['annotations']):
            if len(id)!= prev_len:
                #if j+1<len(d['annotations']):
                #    j+=1
                #changed_frames_d[vid_name] = j
                changed_frames_d[vid_id] = j
                print(f'vid {i}, frames 0,{j} lens: {prev_len},{len(id)}')
                break
    #assert len(dicts) == len(changed_frames_d)
    return changed_frames_d

def find_frames_ids_from_json(dicts,n_frames=2,random_frames=False):
    changed_frames_d={}
    for i,d in enumerate(dicts):
        vid_name = d["file_names"][0].split('/')[-2]
        vid_id = d['video_id'] #check by id
        vid_len = len(d['annotations'])
        if random_frames:
            changed_frames_d[vid_id] = (randrange(vid_len),randrange(vid_len))
        else:
            prev_len = len(d['annotations'][0])
            found=False 
            for j,id in enumerate(d['annotations']):
                if len(id)!= prev_len:
                    changed_frames_d[vid_id] = j
                    print(f'vid {i}, frames 0,{j} lens: {prev_len},{len(id)}')
                    found=True
                    break
            if not found:
                changed_frames_d[vid_id] = 4 #None
                continue

    return changed_frames_d



def overwrite_json(json_file,changed_frames_dict,new_json_file,fraction,class_balance=0,random_frames=False):
    with open(json_file,'r') as f: 
        _json = json.load(f)

    delete_list = set()
    n_classes = len(YTVIS_CATEGORIES_2021)
    n_frames = 6000
    loaded = [0]*n_classes
    print('*******************************')
    #random_gen =np.random.default_rng(42) #random seed
    #for idx,vid in enumerate(_json['videos']):
    
    
    vid_indices = list(range(len(_json['videos'])))
    random.shuffle(vid_indices)
    for idx in vid_indices:
        vid = _json['videos'][idx]
    
        vid_id = vid['id']

        #if random_gen.random() > fraction:
        #    skips.append(idx)    
        #    continue
        #vid_name =  vid["file_names"][0].split('/')[-2]
        #save idx to delete from 'videos' and 'annotations'
        if vid_id not in changed_frames_dict.keys():# or random.random() > fraction:
            delete_list.add(idx)
            continue

        if loaded[_json['annotations'][idx]['category_id']-1] > (n_frames/n_classes) *fraction*class_balance:
            if sum(loaded) > n_frames*fraction:
                delete_list.add(idx)
                continue
        loaded[_json['annotations'][idx]['category_id']-1]+=1
        video = _json['videos'][idx]
        frame_idx = changed_frames_dict[vid_id]
        #take last frame when num classes does not change
        #if frame_idx >= len(video['file_names']): 
        #    frame_idx = len(video['file_names'])-1
        if random_frames:
            video['file_names'] = [video['file_names'][frame_idx[0]],video['file_names'][frame_idx[1]]]
        else:    
            video['file_names'] = [video['file_names'][0],video['file_names'][frame_idx]]
        video['length'] = 2


    #delete_list_ann = set()
    for idx, annotation in enumerate(_json['annotations']):
        vid_id = annotation['video_id']

        if vid_id not in changed_frames_dict.keys():
            continue

        frame_idx = changed_frames_dict[vid_id]
        if random_frames:
            annotation['segmentations'] =[annotation['segmentations'][frame_idx[0]],annotation['segmentations'][frame_idx[1]]]
            annotation['bboxes'] =[annotation['bboxes'][frame_idx[0]],annotation['bboxes'][frame_idx[1]]]
            annotation['areas'] =[annotation['areas'][frame_idx[0]],annotation['areas'][frame_idx[1]]]
        else:
            annotation['segmentations'] =[annotation['segmentations'][0],annotation['segmentations'][frame_idx]]
            annotation['bboxes'] =[annotation['bboxes'][0],annotation['bboxes'][frame_idx]]
            annotation['areas'] =[annotation['areas'][0],annotation['areas'][frame_idx]]
        
    #delete videos with one instance!
    _json['videos'] = [vid for vid in _json['videos'] if vid['id'] not in delete_list]
    _json['annotations'] = [ann for ann in _json['annotations'] if ann['video_id'] not in delete_list]
    
    print('saving: *******************************')
    print(f'class apearance: {loaded}')
    with open(new_json_file, 'w') as f:
        json.dump(_json,f) #ensure_ascii=False ,indent = 4
    return _json



if __name__ == "__main__":
    random_frames=False
    two_frames=True
    fraction = 0.5
    str_frac = str(fraction) if fraction!=1 else ""
    class_balance = 0.7

    json_file = "./datasets/armbench/train.json"
    new_json_file = f"./datasets/armbench/train.json/train_labeled_{str_frac}.json"
    anno_dir = "./datasets/armbench/annotations/"
    image_root = "./datasets/armbench/images/" #.jpg
    logger = setup_logger(name=__name__)
    armbanch_ds = CocoDataset(base_path="./datasets/armbench",convert_to_tami_convention=False)
    
    #dicts = load_ytvis_json(json_file, image_root,old_load=True)
    logger.info("Done loading {} samples.".format(len(dicts)))




if __name__ == "__main_ytvis__":
    random_frames=False
    random_str = "_RANDOM" if random_frames else ""
    two_frames=True
    fraction = 0.5
    str_frac = str(fraction) if fraction!=1 else ""
    class_balance = 0.7 #[0-1],means what portion out of the fraction will keep class balance 
    json_file = "./datasets/ytvis_2021/annotations/instances_train_sub.json"
    new_json_file = f"./datasets/ytvis_2021/annotations/instances_train_sub_2f_{str_frac}{random_str}.json"
    image_root = "./datasets/ytvis_2021/train/JPEGImages"


    #saving with annotation for visualization
    save_annoteted = False #save to dirname 
    dirname = "data-visualization"
    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("ytvis_2021_train")

    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2021_train",old_load=True)
    logger.info("Done loading {} samples.".format(len(dicts)))

    ##activate when two sample is loaded
    #changed_frames_dict = _list_to_dict(dicts)
    changed_frames_dict = find_frames_ids_from_json(dicts,random_frames=random_frames)


    _ = overwrite_json(json_file,changed_frames_dict, new_json_file,fraction,class_balance,random_frames=random_frames)    
    
    #new indices! #todo: move inside save_annoteted
    dicts = load_ytvis_json(new_json_file, image_root, dataset_name="ytvis_2021_train")
    
    if save_annoteted:
        
        os.makedirs(dirname, exist_ok=True)
        #exit(0)
        def extract_frame_dic(dic, frame_idx):
            import copy
            frame_dic = copy.deepcopy(dic)
            annos = frame_dic.get("annotations", None)
            if annos:
                frame_dic["annotations"] = annos[frame_idx]

            return frame_dic

        for d in dicts:
            vid_name = d["file_names"][0].split('/')[-2]
            os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
            for idx, file_name in enumerate(d["file_names"]):
                img = np.array(Image.open(file_name))
                visualizer = Visualizer(img, metadata=meta)
                vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
                fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
                vis.save(fpath)
            print(f'done with {vid_name}')
#ytvis-data-vis/0f17fa6fcb