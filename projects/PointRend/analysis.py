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
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/user/text_inr/detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
# cfg.MODEL.WEIGHTS = "/home/user/text_inr/detectron2/projects/PointRend/output_list/output_totaltext/model_0029999.pth"
cfg.MODEL.WEIGHTS = "/home/user/text_inr/pointrend2/pointrend/projects/PointRend/output_attention/model_0019999.pth"
predictor = DefaultPredictor(cfg)


image_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/test/a00003.jpg"
sdf_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/seg_test3/a00003_maskfg.png"
image_name = os.path.basename(image_path)
im = cv2.imread(os.path.join(image_path))
sdf = cv2.imread(os.path.join(sdf_path), cv2.IMREAD_GRAYSCALE)
outputs = predictor(im,sdf)
outputs = torch.argmax(outputs['sem_seg'],dim=0)
label = np.array(outputs.cpu())
label = np.where(label==1,255,label)
png = Image.fromarray(label.astype(np.uint8)).convert('P')
# textseg
# png.save("/home/user/text_inr/detectron2/projects/PointRend/visual_results/pointrend_total_text/{}.png".format(d["file_name"][-10:].strip(".jpg")))
png.save("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/analysis/final_pred/{}.png".format(image_name[:-4]))