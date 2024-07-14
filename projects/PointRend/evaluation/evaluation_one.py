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

# 설정 파일 로드
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "/home/user/text_inr/pointrend2/pointrend/projects/PointRend/output_attention/model_0019999.pth"

predictor = DefaultPredictor(cfg)

# 단일 이미지 파일 경로 지정
image_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/test/c00243.jpg"
sdf_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/seg_test3/c00243_maskfg.png"

# 이미지 읽기
im = cv2.imread(image_path)
# sdf = np.load(sdf_path)
sdf = cv2.imread(os.path.join(sdf_path), cv2.IMREAD_GRAYSCALE)
# 추론 수행
outputs = predictor(im, sdf)

# 결과 처리
outputs = torch.argmax(outputs['sem_seg'], dim=0)
label = np.array(outputs.cpu()) * 255
png = Image.fromarray(label.astype(np.uint8)).convert('P')

# 결과 저장
output_path = "/home/user/text_inr/pointrend2/pointrend/projects/PointRend/new_quantitative/llm_dir/llm_anal/output.png"
png.save(output_path)