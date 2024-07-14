import os
import natsort
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from skimage.io import imread
from detectron2.structures import BoxMode
import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
from PIL import Image
import os
import natsort
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt

from detectron2.projects import point_rend
from detectron2.projects import deeplab
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import torch
import numpy as np
import PIL
from PIL import Image
from detectron2.utils.visualizer import ColorMode
import natsort

# Define the paths as variables
weight_path = "/home/user/text_inr/pointrend2/SDF-LTSE/projects/PointRend/output/model_0039999.pth" # 디은빋은 weight 파일 경로
config_file = "./projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml" # config 파일 경로, 고정경로로 수정 X
image_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/test" # 이미지 파일 경로
save_path = "/home/user/text_inr/pointrend2/SDF-LTSE/projects/PointRend/new_quantitative/api" # 저장 파일 경로

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file(config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = weight_path
predictor = DefaultPredictor(cfg)

image_list = natsort.natsorted(os.listdir(image_path))

for i, d in enumerate(image_list):
    im = cv2.imread(os.path.join(image_path, d))
    outputs = predictor(im)
    outputs = torch.argmax(outputs['sem_seg'], dim=0)
    label = np.array(outputs.cpu()) * 255
    png = Image.fromarray(label.astype(np.uint8)).convert('P')
    png.save(os.path.join(save_path, f"{image_list[i][:-4]}.png"))
print("done")
