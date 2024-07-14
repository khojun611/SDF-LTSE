from PIL import Image
import numpy as np
import os

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
input_folder = "/home/user/text_inr/detectron2/projects/PointRend/quantitative_results/fusionloss3/lr0.001_40000"  # 입력 이미지가 저장된 폴더 경로
target_folder = "/home/user/text_inr/detectron2/projects/PointRend/datasets/textseg/binary_test"
  # 대상 이미지가 저장된 폴더 경로
evaluate_iou_and_average(input_folder, target_folder)