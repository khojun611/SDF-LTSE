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
cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
# cfg.MODEL.WEIGHTS = "/home/user/text_inr/detectron2/projects/PointRend/output_list/output_totaltext/model_0029999.pth"
cfg.MODEL.WEIGHTS = "/home/user/text_inr/pointrend2/pointrend/projects/PointRend/output/model_0029999.pth"
predictor = DefaultPredictor(cfg)

image_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/test"
image_list = natsort.natsorted(os.listdir(image_path))

sdf_path = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/seg_test"
sdf_list = natsort.natsorted(os.listdir(sdf_path))

for i,d in enumerate(image_list):
    im = cv2.imread(os.path.join(image_path,d))
    sdf = cv2.imread(os.path.join(sdf_path,sdf_list[i]), cv2.IMREAD_GRAYSCALE)
    # sdf = None
    # sdf = np.load(os.path.join(sdf_path,sdf_list[i]))
    outputs = predictor(im,sdf)
    '''
    print(outputs.shape)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # dim=1의 첫 번째 채널(1)에서의 confidence map 선택
    confidence_map = probabilities[:, 1, :, :]

    # confidence map을 numpy 배열로 변환
    confidence_map_np = confidence_map.squeeze().cpu().detach().numpy()

    # 이미지 시각화
    plt.imshow(confidence_map_np, cmap='jet', interpolation='bilinear')
    plt.colorbar()  # 색상 막대 추가
    plt.title("Confidence Map Visualization")
    plt.savefig("/mnt/data/confidence_map_cmap/{}.png".format(image_list[i][:-4]))  # 이미지 파일로 저장
    '''
    outputs = torch.argmax(outputs['sem_seg'],dim=0)
    label = np.array(outputs.cpu())*255
    # label = np.where(label==1,255,label)
    png = Image.fromarray(label.astype(np.uint8)).convert('P')
    # textseg
    # png.save("/home/user/text_inr/detectron2/projects/PointRend/visual_results/pointrend_total_text/{}.png".format(d["file_name"][-10:].strip(".jpg")))
    png.save("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/new_quantitative/0531_coarse/{}.png".format(image_list[i][:-4]))


def load_binary_image(image_path):
    """이미지를 로드하고 이진 형태로 변환합니다."""
    image = Image.open(image_path).convert('1')  # 이진 이미지로 변환
    
    return np.array(image, dtype=np.bool_)  # 수정된 부분

def calculate_iou(image1, image2):
    """두 이진 이미지 간의 IoU를 계산합니다."""
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1*255, image2).sum()
    iou = intersection / union
    return iou

def evaluate_iou_and_average(input_folder, target_folder):
    """입력 폴더와 대상 폴더에 있는 이미지 쌍 간의 IoU를 평가하고 평균을 계산합니다."""
    input_images = os.listdir(input_folder)
    
    iou_scores = []  # 계산된 IoU 점수를 저장할 리스트

    # 입력 이미지 파일에 대응하는 대상 이미지 파일을 찾아 IoU를 계산합니다.
    for input_img_name in input_images:
        # 입력 이미지 이름에서 대응하는 대상 이미지 이름을 생성합니다.
        target_img_name = input_img_name.split('.')[0] + '_maskfg.png'

        # 대상 이미지 파일이 존재하는지 확인합니다.
        target_image_path = os.path.join(target_folder, target_img_name)
        if os.path.exists(target_image_path):
            input_image_path = os.path.join(input_folder, input_img_name)

            input_image = load_binary_image(input_image_path)
            target_image = load_binary_image(target_image_path)

            # IoU 계산
            iou = calculate_iou(input_image, target_image)
            iou_scores.append(iou)  # IoU 점수를 리스트에 추가
            print(f"IoU for {input_img_name} and {target_img_name}: {iou}")
        else:
            print(f"Target image {target_img_name} not found for input image {input_img_name}.")

    # IoU 점수의 평균을 계산합니다.
    if iou_scores:
        average_iou = np.mean(iou_scores)
        print(f"Average IoU: {average_iou}")
    else:
        print("No IoU scores were calculated.")

# 사용 예
input_folder = "/home/user/text_inr/pointrend2/pointrend/projects/PointRend/new_quantitative/0531_coarse"  # 입력 이미지가 저장된 폴더 경로
target_folder = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/binary_test"
  # 대상 이미지가 저장된 폴더 경로
evaluate_iou_and_average(input_folder, target_folder)

