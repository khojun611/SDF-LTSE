from PIL import Image
import numpy as np
import os

def convert_image_to_255(image_path, save_path):
    """이미지를 로드하고 0과 255로 변환한 뒤 저장합니다."""
    # 이미지 로드
    image = Image.open(image_path)
    # PIL 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    
    # 0과 1 값으로 구성된 배열을 0과 255로 스케일링
    img_array_255 = img_array * 255
    
    # 변환된 배열을 이미지로 변환
    img_255 = Image.fromarray(img_array_255.astype(np.uint8))
    
    # 이미지 저장
    img_255.save(save_path)

def process_folder(folder_path):
    """폴더 내의 모든 이미지를 0과 255로 변환합니다."""
    for img_name in os.listdir(folder_path):
        # 원본 이미지 경로
        img_path = os.path.join(folder_path, img_name)
        # 저장할 이미지 경로 (이 예에서는 원본 위치에 덮어쓰기)
        save_path = img_path
        
        # 이미지 변환 및 저장
        convert_image_to_255(img_path, save_path)

# 폴더 경로 설정
folder_path = "/home/user/text_inr/detectron2/projects/PointRend/quantitative_results/fusionloss3/lr0.001_40000"
process_folder(folder_path)