# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from .point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sample_sdf_map_points
)
from .point_head import build_point_head
# First we define a simple function to help us plot the intermediate representations.
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import pdb



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes, H, W], 여기서 num_classes는 2
        # targets: [batch_size, H, W]

        # 무시해야 할 위치의 마스크를 생성
        mask = (targets != self.ignore_index).float()
        targets = (targets * mask).long()  # 무시할 부분을 0으로 설정

        # 클래스에 대한 확률을 구하기 위한 softmax
        probs = F.softmax(logits, dim=1) 
        # 클래스 1에 대한 확률
        probs_at_class1 = probs[:, 1, :] 

        # 이진 분류 문제의 확률과 대상 레이블을 사용하여 focal loss 계산
        pt = torch.where(targets == 1, probs_at_class1, 1 - probs_at_class1)
        targets = targets.float()
        
        ce_loss = F.binary_cross_entropy(probs_at_class1, targets, reduction='none')
        focal_term = (1 - pt).pow(self.gamma)
        
        loss = self.alpha * focal_term * ce_loss

        # 마스크를 사용하여 손실에서 무시할 위치를 0으로 설정
        loss = loss * mask

        if self.reduction == 'mean':
            return loss.sum() / mask.sum()  # 실제 계산에 포함된 요소만으로 평균
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_value=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_value = ignore_value

    def forward(self, logits, targets):
        # 다중 채널 로짓을 확률로 변환
        probs = F.softmax(logits, dim=1)
        
        # 관심 클래스(전경 클래스)의 확률 사용
        probs = probs[:, 1, :]
        
        # 라벨과 예측 텐서를 평평하게 만듭니다.
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        # ignore value에 해당하는 위치의 mask를 생성합니다.
        mask = (targets != self.ignore_value).float()
        
        # Masking 처리
        probs = probs * mask
        targets = targets * mask
        
        intersection = (probs * targets).sum(1)
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        
        return 1 - dice_coeff.mean()

def plot_mask(mask, title="", point_coords=None, figsize=10, point_marker_size=5,steps=None):
    '''
    Simple plotting tool to show intermediate mask predictions and points 
    where PointRend is applied.
    
    Args:
        mask (Tensor): mask prediction of shape HxW
        title (str): title for the plot
        point_coords ((Tensor, Tensor)): x and y point coordinates
        figsize (int): size of the figure to plot
        point_marker_size (int): marker size for points
    '''

    H, W = mask.shape
    plt.figure(figsize=(figsize, figsize))
    if title:
        title += ", "
    plt.title("{}resolution {}x{}".format(title, H, W), fontsize=30)
    plt.ylabel(H, fontsize=30)
    plt.xlabel(W, fontsize=30)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(mask, interpolation="nearest", cmap=plt.get_cmap('gray'))
    if point_coords is not None:
        plt.scatter(x=point_coords[0], y=point_coords[1], color="red", s=point_marker_size, clip_on=True) 
    plt.xlim(-0.5, W - 0.5)
    plt.ylim(H - 0.5, - 0.5)
    plt.savefig("/home/user/text_inr/detectron2/projects/PointRend/lr_images/point_sampling/test{}.png".format(steps))

def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    # print("sem_seg_logit",sem_seg_logits)
    # print("top2_scores shape",top2_scores.shape)
    # print("top2_scores[0]",top2_scores[:,0])
    # print("top2_socres[1]",top2_scores[:,1])
    # print("uncertain map shape",(top2_scores[:, 1] - top2_scores[:, 0]).shape)
    # print("uncertatinty map",(top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1))
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)

def transform_to_probability(matrix, dim=1, temperature=1.0):
    scaled_matrix = matrix / temperature
    probabilities = F.softmax(scaled_matrix, dim=dim)
    return probabilities

def normalize(feature_map):
    feature_map_min = feature_map.min()
    feature_map_max = feature_map.max()
    normalized_feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)
    return normalized_feature_map


def freeze_model(model):
    for param in model.parameters():
        param.require_grad = False

@SEM_SEG_HEADS_REGISTRY.register()
class PointRendSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.batch_size = int(int(cfg.SOLVER.IMS_PER_BATCH)/2)
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # print("batch size",self.batch_size)
        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
            cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME
        )(cfg, input_shape)
        self._init_point_head(cfg, input_shape)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        # self.predictor = Conv2d(self.batch_size, self.num_classes, kernel_size=1, stride=1, padding=0)


    def _init_point_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        # fmt: off
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        feature_channels             = {k: v.channels for k, v in input_shape.items()}
        self.in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # self.upsample_x2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1).to('cuda')
        # self.upsample_x4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1).to('cuda')
        
        # self.sdf_layer = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1).to('cuda')
        # self.sdf_layer2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1).to('cuda')
        # self.sdf_head = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1).to('cuda')
        # self.relu = F.relu
        
        
        # self.batch_norm1 = torch.nn.BatchNorm2d(128)
        # self.batch_norm2 = torch.nn.BatchNorm2d(1)
        
        """
        self.sdf_layer = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        # self.sdf_layer2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.sdf_head = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        """
        # self.cls_layer = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0,activation=F.relu).to('cuda')
        # self.cls_layer2 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0,activation=F.relu).to('cuda')
        # fmt: on
        
        

        in_channels = int(np.sum([feature_channels[f] for f in self.in_features]))
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def _get_point_logits(self, coarse_features,
                          features,point_coords,coarse_sem_seg_logits):
        return self.point_head(coarse_features,
                          features,point_coords,coarse_sem_seg_logits)
    
    # freeze_model(self.coarse_sem_Seg)
    def calculate_sdf_batch(self, tensor):
        batch_sdf = []
        for i in range(tensor.size(0)):
            image_np = tensor[i].cpu().numpy()

            # 물체 외부와 내부에 대한 거리 변환을 계산
            if np.max(image_np) > 0:
                dist_outside = distance_transform_edt(image_np)  # 물체 바깥쪽 거리
                dist_inside = distance_transform_edt(1 - image_np)  # 물체 안쪽 거리
                sdf = dist_inside - dist_outside  # SDF 계산, 부호 반전
            else:
                # 이미지 내에 물체가 없는 경우, 모두 -100으로 처리
                sdf = -100 * np.ones_like(image_np)

            # 정규화 단계 없이 바로 SDF 맵 추가
            batch_sdf.append(sdf)

        # 리스트의 NumPy 배열들을 스택하여 새 NumPy 배열 생성
        batch_sdf_np = np.stack(batch_sdf, axis=0)
        
        # NumPy 배열을 PyTorch 텐서로 변환
        batch_sdf_tensor = torch.from_numpy(batch_sdf_np).float()

        return batch_sdf_tensor
    
    def calculate_sdf_normalization(self, tensor):
        batch_sdf = []
        for i in range(tensor.size(0)):
            image_np = tensor[i].cpu().numpy()

            if np.max(image_np) > 0:
                dist_outside = distance_transform_edt(image_np)
                dist_inside = distance_transform_edt(1 - image_np)
                # 물체의 내부와 외부 거리 값의 부호를 뒤집습니다.
                sdf = dist_outside - dist_inside
            else:
                # 이미지 내에 물체가 없는 경우, 모두 0으로 처리
                sdf = np.zeros_like(image_np) - 0.99

            # SDF 맵 정규화
            # 양수 부분을 [0, 1]로 정규화
            sdf_positive = np.clip(sdf, 0, None)
            if sdf_positive.max() > 0:
                sdf_positive /= sdf_positive.max()

            # 음수 부분을 [-1, 0]로 정규화
            sdf_negative = np.clip(sdf, None, 0)
            if sdf_negative.min() < 0:
                sdf_negative /= np.abs(sdf_negative.min())

            sdf_normalized = sdf_positive + sdf_negative

            batch_sdf.append(sdf_normalized)

        batch_sdf_np = np.stack(batch_sdf, axis=0)
        batch_sdf_tensor = torch.from_numpy(batch_sdf_np).float()

        return batch_sdf_tensor

    def calculate_sdf_normalization2(self, tensor):
        batch_sdf = []
        for i in range(tensor.size(0)):
            image_np = tensor[i].cpu().numpy()

            # 물체 외부와 내부에 대한 거리 변환을 계산
            if np.max(image_np) > 0:
                dist_outside = distance_transform_edt(image_np)
                dist_inside = distance_transform_edt(1 - image_np)
                sdf = dist_outside - dist_inside
            else:
                # 이미지 내에 물체가 없는 경우, 모두 0으로 처리
                sdf = np.zeros_like(image_np)

            # SDF 맵 정규화
            # 양수 부분을 [0, 1]로 정규화
            sdf_positive = np.clip(sdf, 0, None)
            if sdf_positive.max() > 0:
                sdf_positive /= sdf_positive.max()

            # 음수 부분을 [-1, 0]로 정규화
            sdf_negative = np.clip(sdf, None, 0)
            if sdf_negative.min() < 0:
                sdf_negative /= np.abs(sdf_negative.min())

            sdf_normalized = sdf_positive + sdf_negative

            batch_sdf.append(sdf_normalized)

        batch_sdf_np = np.stack(batch_sdf, axis=0)
        batch_sdf_tensor = torch.from_numpy(batch_sdf_np).float()

        return batch_sdf_tensor
    
    def calculate_sdf_normalization3(self,tensor):
        batch_sdf = []
        for i in range(tensor.size(0)):
            image_np = tensor[i].cpu().numpy()

            # 물체 외부와 내부에 대한 거리 변환을 계산
            if np.max(image_np) > 0:
                dist_outside = distance_transform_edt(image_np)
                dist_inside = distance_transform_edt(1 - image_np)
                sdf = dist_outside - dist_inside
            else:
                # 이미지 내에 물체가 없는 경우, 모두 0으로 처리
                sdf = np.zeros_like(image_np)

            # SDF 맵 정규화
            # 양수 부분을 [0, 1]로 정규화
            sdf_positive = np.clip(sdf, 0, None)
            if sdf_positive.max() > 0:
                sdf_positive /= sdf_positive.max()

            # 음수 부분을 [-1, 0]로 정규화
            sdf_negative = np.clip(sdf, None, 0)
            if sdf_negative.min() < 0:
                sdf_negative /= np.abs(sdf_negative.min())

            sdf_normalized = sdf_positive + sdf_negative

            # 원본 SDF 맵에서 0이었던 부분을 정확히 0으로 설정
            sdf_normalized = np.where(sdf == 0, 0, sdf_normalized)

            batch_sdf.append(sdf_normalized)

        batch_sdf_np = np.stack(batch_sdf, axis=0)
        batch_sdf_tensor = torch.from_numpy(batch_sdf_np).float()

        return batch_sdf_tensor

    def sample_closest_points_per_image(self,sdf_map, n=2048):
        batchsize, H, W = sdf_map.shape
        # 결과를 저장할 텐서 초기화
        coordinates_batch = torch.zeros((batchsize, n, 2), dtype=torch.float)
        # print(batchsize)
        for i in range(batchsize):
            # i번째 이미지의 SDF 맵 절대값 계산
            abs_sdf = sdf_map[i].abs()
            
            # 절대값을 펼쳐서 정렬하고, 상위 n개 인덱스 선택
            flat_abs_sdf = abs_sdf.view(-1)
            sorted_indices = flat_abs_sdf.argsort()
            top_n_indices = sorted_indices[:n]
            
            # 인덱스를 x, y 좌표로 변환
            y_indices = torch.div(top_n_indices, W, rounding_mode='floor')
            x_indices = top_n_indices % W
            
            # 좌표 병합
            coordinates = torch.stack((x_indices, y_indices), dim=1)
            
            # 결과 저장
            
            coordinates_batch[i] = coordinates
        
        return coordinates_batch

    # 예제 데이터 생성
    # batchsize, H, W = 3, 100, 100 # 이미지 크기를 크게 설정하여 충분한 포인트가 있도록 함

    # 임의의 SDF 맵 생성
    
    def forward(self, features, targets=None, sdf=None):
        # print("features key",features.keys())
        # coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features)
        
        '''
        probs = F.softmax(coarse_sem_seg_logits.cpu(),dim=1)
        
        plt.figure(figsize=(10,5))
        
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.imshow(probs[0, 0, :, :].detach().numpy(), cmap='jet')
        plt.title('Class 0 Probability Map')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(probs[0, 1, :, :].detach().numpy(), cmap='jet')
        plt.title('Class 1 Probability Map')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('/home/user/text_inr/detectron2/projects/PointRend/lr_images/coarse_output/probability_map/probability_maps3.png', dpi=300)
        '''

        # print("coarse_sem_seg_logits", coarse_sem_seg_logits)
        

        if self.training:

            p2 = features["p2"].size()
            # print("p2 shape",p2.shape)
            # print("target shape",targets.shape)
            # print("sdf shape",sdf.shape)
            # print("target.shape",targets.shape)
            sdf = self.calculate_sdf_normalization3(targets).cuda()
            # print("sdf 2 shape",sdf.shape)
            
            # attention2 = F.interpolate(sdf.unsqueeze(1), 
            #    size=p2.size()[-2:], mode='bilinear', align_corners=False)
            
            attention2 = F.interpolate(sdf.unsqueeze(1).float(),
                    size=p2[-2:], mode='bilinear', align_corners=True)
            # print("sdf[0]",sdf[0])

            # print("sdf[0]",sdf[0])
            
            coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features,attention2)
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets)
            sdf = torch.argmax(coarse_sem_seg_logits, dim=1)
            sdf = self.calculate_sdf_normalization3(sdf).cuda()
            # print("sdf 2 shape",sdf.shape)
            
            # attention2 = F.interpolate(sdf.unsqueeze(1), 
            #    size=p2.size()[-2:], mode='bilinear', align_corners=False)
            
            attention2 = F.interpolate(sdf.unsqueeze(1).float(),
                    size=p2[-2:], mode='bilinear', align_corners=True)
            coarse_sem_seg_logits2 = self.coarse_sem_seg_head.layers2(features,attention2)
            losses["loss_sem_seg2"] = self.coarse_sem_seg_head.losses2(coarse_sem_seg_logits2, targets)
            
            # print("coarse_sem_seg_logits",coarse_sem_seg_logits.shape)
            # print("targets",targets.shape)
            # print("loss",losses)
            num_points = 6000
            with torch.no_grad():
                # point_coords = self.sample_closest_points_per_image(sdf_map=sdf_feature,n=self.train_num_points).cuda()
                
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_sem_seg_logits,
                    calculate_uncertainty,
                    num_points,# self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                
                
                
                
                
                # point_coords = self.sample_closest_points_per_image(sdf_map=sdf_feature,n=self.train_num_points).cuda().float()
                
            # print("point_coords shape",point_coords.shape)
            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            

            fine_grained_features = cat(
                [
                    point_sample(features[in_feature], point_coords, align_corners=False)
                    for in_feature in self.in_features
                ],
                dim=1,
            )
            point_logits = self._get_point_logits(coarse_features,
                          features,point_coords,coarse_sem_seg_logits)
            # self.point_head(fine_grained_features, coarse_features,features,point_coords,coarse_sem_seg_logits) # coarse map 전달
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                ) 
                .squeeze(1)
                .to(torch.long)
            )
            # print("point_coords shape",point_coords.shape)
            # print("point_logits.shape",point_logits.shape)
            
            cs_loss = F.cross_entropy(
                point_logits, point_targets, reduction="mean", ignore_index=self.ignore_value
            )
            # dice_loss = self.dice_loss(point_logits, point_targets)
            # focal_loss =self.focal_loss(point_logits,point_targets)
            
            
            total_loss = cs_loss
            
            
            
            losses["loss_sem_seg_point"] = total_loss 
            
            # losses["loss_sdf_l1_loss"]= sdf_loss 
            
            return None, losses
        else:
            
            p2 = features["p2"]
            # data = torch.randn(1, 256, H, W)  # H와 W는 실제 차원에 맞게 설정
            # sdf = self.calculate_sdf_normalization3(sdf).cuda()
            
            

            attention2 = None
            coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features,attention2)
            
            sdf = torch.argmax(coarse_sem_seg_logits, dim=1)
            sdf = self.calculate_sdf_normalization3(sdf).cuda()
            # print("sdf shape", sdf.shape)
            attention2 = F.interpolate(sdf.unsqueeze(0), 
                size=p2.size()[-2:], mode='bilinear', align_corners=False)
            coarse_sem_seg_logits2 = self.coarse_sem_seg_head.layers2(features,attention2)
            # print("attention2",attention2.size())
            
            sem_seg_logits = coarse_sem_seg_logits.clone()
            
            
            """
            outputs = sem_seg_logits = F.interpolate(
                    sem_seg_logits, scale_factor=4, mode="bilinear", align_corners=False
                )
            # print("sem_seg_logits shape",sem_seg_logits.shape)
            outputs = torch.argmax(outputs[0],dim=0)
            # print("output shape",outputs.shape)
            label = np.array(outputs.cpu())
            label = np.where(label==1,255,label)
            png = Image.fromarray(label.astype(np.uint8)).convert('P')
            png.save("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/analysis/coarse_pred/c01757.png")
            """
            
            # print('sem_seg_origin',sem_seg_logits.shape)
            for i in range(self.subdivision_steps):
                sem_seg_logits = F.interpolate(
                    sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
                )

                

                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                # 여기 아랫부분이 uncertainty map 생성
                
                
                
                
                # point_indices, point_coords = sample_sdf_map_points(attention2,num_points)
                point_indices, point_coords =get_uncertain_point_coords_on_grid(
                   uncertainty_map,self.subdivision_num_points
                    )
                
                # torch.save(point_coords.cpu(),"/home/user/text_inr/detectron2/projects/PointRend/lr_images/yesim/point/a00012.pt")
                fine_grained_features = cat(
                    [
                        point_sample(features[in_feature], point_coords, align_corners=False)
                        for in_feature in self.in_features
                    ]
                )
                fine_grained_features = fine_grained_features.permute(0,-1,1)
                

                coarse_features = point_sample(
                    coarse_sem_seg_logits, point_coords, align_corners=False
                )
                point_logits = self._get_point_logits(coarse_features,
                          features,point_coords,coarse_sem_seg_logits)
                # print("point_logits",point_logits)
                # print("point_logits",point_logits)
                
                # self.point_head(fine_grained_features, coarse_features,features,coarse_sem_seg_logits,features,point_coords)

                # put sem seg point predictions to the right places on the upsampled grid.
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
                
                
                # print("point_indices shape",point_indices.shape)
                # print("point_indices",point_indices)
                # print("point_coords shape",point_coords.shape)
                # print("sem seg logit shape",sem_seg_logits.shape)
                # print("sem_seg_logit",sem_seg_logits)
            """   
            plot_mask(
                sem_seg_logits.to("cpu"),
                title = "Sampled points over the coarse prediction",
                point_coords=(
                W * point_coords[:,0].to("cpu") - 0.5,
                H * point_coords[:,1].to("cpu") - 0.5),
                
                point_marker_size=50,
                steps=i
                )
            """
                # print("sem_seg_logits2",sem_seg_logits.shape)
                
            return sem_seg_logits, {}
        