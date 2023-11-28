import os
import json
import numpy as np
import shutil


PART = 1/10
PART_STR = "1_10"
NUM_SAMPLES = int(44253*PART)

NAME_JSON = 'train.json'
PATH_ROOT = os.path.dirname(__file__)

class Key:
    ID = 'id'
    IMAGE_ID = 'image_id'
    CATEGORIES = 'categories'
    IMAGES = 'images'
    ANNOTATIONS = 'annotations'
    NUM_IMAGES = 'num_images'
    NUM_ANNOTATIONS = 'num_annotations'
    FILE_NAME = 'file_name'



PATH_SOURCE = PATH_ROOT
PATH_SOURCE_JSON = os.path.join(PATH_SOURCE, NAME_JSON)
PATH_SOURCE_IMAGES = os.path.join(PATH_SOURCE, Key.IMAGES)
PATH_SOURCE_ANNOTATIONS = os.path.join(PATH_SOURCE, Key.ANNOTATIONS)

PATH_TARGET = PATH_ROOT #os.path.join(PATH_ROOT, f'armbench_{PART_STR}-mix-object-tote')
PATH_TARGET_JSON = os.path.join(PATH_TARGET, f'{PART_STR}_{NAME_JSON}')
PATH_TARGET_IMAGES = os.path.join(PATH_TARGET, Key.IMAGES)
PATH_TARGET_ANNOTATIONS = os.path.join(PATH_TARGET, Key.ANNOTATIONS)


def create_metadata_subset(path):
    with open (path, 'r') as target:
        data = json.load(target)

    metadata_subset = {
        Key.CATEGORIES: data[Key.CATEGORIES],
        Key.NUM_IMAGES: NUM_SAMPLES
        }

    images = data[Key.IMAGES]
    images_subset = np.random.choice(images, NUM_SAMPLES)
    metadata_subset[Key.IMAGES] = list(images_subset)

    image_ids = set()
    files_images = []
    files_annotations = []

    suffix = '.json'
    for image in images_subset:
        image_ids.add(image[Key.ID])
        name_image = image[Key.FILE_NAME]
        files_images.append(name_image)

        file_annotation = name_image.split('.')[0]
        files_annotations.append(file_annotation + suffix)

    annotations = []
    for annotation in data[Key.ANNOTATIONS]:
        if annotation[Key.IMAGE_ID] in image_ids:
            annotations.append(annotation)
    
    metadata_subset[Key.ANNOTATIONS] = annotations
    metadata_subset[Key.NUM_ANNOTATIONS] = len(annotations)

    return metadata_subset, files_images, files_annotations

def copy_selection_to(path_source, selections, path_target):
    os.makedirs(path_target, exist_ok=True)

    print(f'copy {path_source} to {path_target}')
    total = len(selections)
    for i, selection in enumerate(selections, start=1):
        source = os.path.join(path_source, selection)
        target = os.path.join(path_target, selection)
        print(f'copy {i}/{total}')
        shutil.copyfile(source, target)

def write_metadata(metadata, path_target, target_name):
    os.makedirs(path_target, exist_ok=True)

    path_target_name = os.path.join(path_target, target_name)
    with open(path_target_name, 'w') as outfile:
        json.dump(metadata, outfile)

if __name__ == "__main__":
    subset = create_metadata_subset(PATH_SOURCE_JSON)
    metadata, files_images, files_annotations = subset
    print(len(files_images))
    write_metadata(metadata, PATH_TARGET, f'{PART_STR}_train.json')
    
    # copy_selection_to(PATH_SOURCE_IMAGES, files_images, PATH_TARGET_IMAGES)
    # copy_selection_to(PATH_SOURCE_ANNOTATIONS, files_annotations, PATH_TARGET_ANNOTATIONS)