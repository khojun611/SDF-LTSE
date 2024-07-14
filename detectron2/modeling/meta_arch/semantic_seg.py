# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import Backbone, build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = [
    "SemanticSegmentor",
    "SEM_SEG_HEADS_REGISTRY",
    "SemSegFPNHead",
    "build_sem_seg_head",
]


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes, H, W]
        # targets: [batch_size, H, W]
        
        # 무시해야 할 위치의 마스크 생성
        mask = (targets != self.ignore_index)
        targets = targets[mask]  # 무시할 부분을 계산에서 제외
        
        # logits에서 무시할 부분을 계산에서 제외
        logits_at_class1 = logits[:, 1, :, :][mask]
        
        # 로짓에 대한 sigmoid 적용 및 binary_cross_entropy 계산
        ce_loss = F.binary_cross_entropy_with_logits(logits_at_class1, targets.float(), reduction='none')
        
        # 확률 pt 계산
        probs_at_class1 = torch.sigmoid(logits_at_class1)
        pt = torch.where(targets == 1, probs_at_class1, 1 - probs_at_class1)
        
        # focal term 계산
        focal_term = (1 - pt).pow(self.gamma)
        
        # 최종 focal loss 계산
        loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
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
        probs = probs[:, 1, :, :]
        
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

    
@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.


        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor that represents the
              per-pixel segmentation prediced by the head.
              The prediction has shape KxHxW that represents the logits of
              each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            # print("target",targets)
            targets = ImageList.from_tensors(
                targets,
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            targets = None
        if "sdf_map" in batched_inputs[0]:
            sdf = [[x["sdf_map"].to(self.device) for x in batched_inputs]]
            # print("sdf",sdf)
            sdf = ImageList.from_tensors(
                sdf[0],
                self.backbone.size_divisibility,
                self.sem_seg_head.ignore_value,
                self.backbone.padding_constraints,
            ).tensor
        else:
            sdf = None

        results, losses = self.sem_seg_head(features, targets,sdf)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        if not len(input_shape):
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape]

        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

        # Original decoder
        self.scale_heads = []
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)  

        # New independent decoder
        self.scale_heads2 = []
        for in_feature, stride, channels in zip(
            self.in_features, feature_strides, feature_channels
        ):
            head_ops2 = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops2.append(conv)
                if stride != self.common_stride:
                    head_ops2.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads2.append(nn.Sequential(*head_ops2))
            self.add_module(f"{in_feature}_2", self.scale_heads2[-1])
        self.predictor2 = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor2)
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
        }

    def forward(self,features, targets=None):
        
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        if self.training:
            print("rurushu")
            
            return None, self.losses(x, targets)
        else:
            print("lulushu bibritania")
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}  

    def layers(self, features, attention):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        
        x = self.predictor(x)
        return x

    def layers2(self, features, attention):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads2[i](features[f])
            else:
                x = x + self.scale_heads2[i](features[f])
        x = x * attention
        x = self.predictor2(x)
        return x
        

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        # print("predictions shape",predictions.shape)
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=True,
        )
        # print("targets",targets.shape)
        # print("predictions.shape",predictions.shape)
        # print("targets shape", targets.shape)
        # unique_values = torch.unique(targets)
        # print("unique target", unique_values)
        # print("len unique value",len(unique_values))
        
        cs_loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        
        
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        total_loss =  dice_loss + focal_loss + cs_loss
        # total_loss = cs_loss
        # total_loss = cs_loss
        losses = {"loss_sem_seg": total_loss}
        return losses
    
    def losses2(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=True,
        )

        cs_loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )

        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)

        total_loss = dice_loss + focal_loss + cs_loss
        return total_loss
