#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
import numpy as np
from itertools import groupby
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
from PIL import Image
import os
import natsort
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_textseg_sem(img_dir, mask_dir, sdf_dir):
    img_list = natsort.natsorted(os.listdir(img_dir))
    dataset_dicts = []
    sem_seg_file_list = natsort.natsorted(os.listdir(mask_dir))
    sdf_file_list = natsort.natsorted(os.listdir(sdf_dir))
    for idx, v in  enumerate(img_list):
        record = {}
        filename = os.path.join(img_dir, v)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        sem_seg_file_name = os.path.join(mask_dir,sem_seg_file_list[idx])
        sdf_map_file_name = os.path.join(sdf_dir,sdf_file_list[idx])

        # print("sdf map",sdf_map_file_name)
        record["sem_seg_file_name"] = sem_seg_file_name
        record["sdf_map_file_name"] = sdf_map_file_name
        dataset_dicts.append(record)

    return dataset_dicts
"""
for d in ["train"]:
    DatasetCatalog.register("Textseg_sem_"+d,lambda d=d: get_textseg_sem("/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/train",
                                                            "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/alpha_train_mask"))
    MetadataCatalog.get("Textseg_sem_"+d).set(stuff_classes=["Background",'a','b','c','d','e','f','g','h','i','j','k','l','m',
                                                             'n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C',
                                                             'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',
                                                             'V','W','X','Y','Z','!','@','#','$','%','^','&','*','(',')','1','2',
                                                             '3','4','5','6','7','8','9','0'])
text_metadata = MetadataCatalog.get("Textseg_sem_train")
"""

"""
for d in ["train"]:
    DatasetCatalog.register("Textseg_sem_"+d,lambda d=d: get_textseg_sem("/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/train",
                                                            "/home/user/text_inr/detectron2/projects/PointRend/datasets/new_alphabet_textseg/train2"))
    MetadataCatalog.get("Textseg_sem_"+d).set(stuff_classes=["Background",'a','b','c','e','f','g','h','i','j','k','l','m',
                                                             'n','o','p','u','v','w','x','z','A','B','C',
                                                             'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',
                                                             'V','W','X','Y','Z','Others'])
text_metadata = MetadataCatalog.get("Textseg_sem_train")
"""


for d in ["train"]:
    DatasetCatalog.register("Textseg_sem_"+d,lambda d=d: get_textseg_sem("/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/train",
                                                            "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/seg_train01",
                                                            "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/seg_train01"))
    MetadataCatalog.get("Textseg_sem_"+d).set(stuff_classes=["Background","Text"])
text_metadata = MetadataCatalog.get("Textseg_sem_train")

"""
for d in ["train"]:
    DatasetCatalog.register("Textseg_sem_"+d,lambda d=d: get_textseg_sem("/home/user/text_inr/detectron2/projects/PointRend/datasets/pseudo_dataset2/image",
                                                            "/home/user/text_inr/detectron2/projects/PointRend/datasets/pseudo_dataset2/label"))
    MetadataCatalog.get("Textseg_sem_"+d).set(stuff_classes=["Background","Text"])
text_metadata = MetadataCatalog.get("Textseg_sem_train")
"""

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs

def get_textseg(img_dir, mask_dir):
    
    img_list = natsort.natsorted(os.listdir(img_dir))
    dataset_dicts = {}
    images = []
    annotations = []
    mask_list = natsort.natsorted(os.listdir(mask_dir))
    for idx, v in enumerate(img_list):
        record = {}
        # print(os.path.join(img_dir, v))

        filename = os.path.join(img_dir, v)
        #print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        # images = []
        images_dict = {

            "file_name": filename,
            "id": idx,
            "height": height,
            "width": width
        }

        images.append(images_dict)

        # images["file_name"] = filename
        # images["image_id"] = idx
        # images["height"] = height
        # images["width"] = width

        pixel = Image.open(mask_dir + mask_list[idx])
        pixel = np.array(pixel)
        # print("pixel size", pixel.shape)
        pixel = np.where(pixel==100,1,pixel)
        pixel = np.where(pixel==200,0,pixel)
        pixel = np.where(pixel==255,0,pixel)
    

        fortran_ground_truth_binary_mask = np.asfortranarray(pixel)
        # print("fortran_ground_truth_binary_mask size: ", fortran_ground_truth_binary_mask.shape)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        # print(encoded_ground_truth)
        # print("fortran_ground_truth_binary_mask",fortran_ground_truth_binary_mask)
        # encoded_ground_truth = mask.encode(np.asarray(pixel, order="F"))
        # print("encoded_ground_truth",encoded_ground_truth)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = binary_mask_to_rle(fortran_ground_truth_binary_mask)
        # contours = encoded_ground_truth
        # contours = measure.find_contours(pixel, 0.5)
        # print(contours)
        # contours = mask.encode(np.asarray(pixel, order="F"))

        # objs = []
        annotations_dict = {
            "bbox": ground_truth_bounding_box.tolist(),
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": contours,
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": idx,
            # "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": idx
        }

        
        # obj["segmentation"].append(contours)
        # contour = np.flip(contours)
        # segmentation = contour.ravel().tolist()
        # obj["segmentation"].append(segmentation)
        # for contour in contours:
        #     contour = np.flip(contour, axis=1)
        #     segmentation = contour.ravel().tolist()
        #     annotations_dict["segmentation"].append(segmentation)
        
         
        annotations.append(annotations_dict)
        
        categories_dict = [{
            "supercategory": "text",
            "id": 1,
            "name": "text"

        }]
        # dataset_dicts.append(record)
    dataset_dicts["images"] = images
    dataset_dicts["annotations"] = annotations
    dataset_dicts["categories"] = categories_dict
    return dataset_dicts



def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
"""
register_coco_instances("TextSeg_train", {},"/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/json_list/textseg_train.json","/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/train/")
MetadataCatalog.get("TextSeg_train").thing_classes = ["text"]
register_coco_instances("TextSeg_valid", {},"/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/json_list/textseg_valid.json","/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/valid/")
MetadataCatalog.get("TextSeg_valid").thing_classes = ["text"]
register_coco_instances("TextSeg_test", {},"/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/json_list/textseg_test.json","/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/test/")
MetadataCatalog.get("TextSeg_test").thing_classes = ["text"]
"""
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg




def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


