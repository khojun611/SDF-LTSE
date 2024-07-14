# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

def transform_to_probability(matrix, dim=1, temperature=1.0):
    scaled_matrix = matrix / temperature
    probabilities = F.softmax(scaled_matrix, dim=dim)
    return probabilities

def roi_mask_point_loss(mask_logits, instances, point_labels):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)

        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

    gt_mask_logits = point_labels
    point_ignores = point_labels == -1
    if gt_mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0

    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
        # print("mask_logits",mask_logits)
        # print("mask_logit shape",mask_logits.shape)
        # mask_logits = F.interpolate(mask_logtis, size = (224,224), mode="bilinear") [N,224^2]
        # mask_logits = LR_image + mask_logits [N,224^2]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]
        # print("mask_logits",mask_logits)
        # print("mask_logit shape",mask_logits.shape) # [N,14^2]
        # mask_logits = F.interpolate(mask_logtis, size = (224,224), mode="bilinear") [N,224^2]
        # mask_logits = LR_image + mask_logits [N,224^2]
        

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.to(dtype=torch.uint8)
    mask_accurate = mask_accurate[~point_ignores]
    mask_accuracy = mask_accurate.nonzero().size(0) / max(mask_accurate.numel(), 1.0)
    get_event_storage().put_scalar("point/accuracy", mask_accuracy)

    # print("gt_mask_logtis shape",gt_mask_logits.shape)
    # print("gt_mask_logits",gt_mask_logits)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), weight=~point_ignores, reduction="mean"
    )
    
    
    
    return point_loss

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@POINT_HEAD_REGISTRY.register()
class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        self.in_features            = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.input_channels         = input_shape.channels
        self.input_height           = input_shape.height
        self.input_width            = input_shape.width
        hidden_dim                  = 256
        self.coef                   = nn.Conv2d(self.input_channels, hidden_dim, 3, padding=1)
        self.freq                   = nn.Conv2d(self.input_channels, hidden_dim, 3, padding=1)
        self.phase                  = nn.Linear(2, hidden_dim//2, bias=False)
        # fmt: on
        fc_dim_in                   = self.input_channels # + num_classes
        self.imnet                  = MLP(in_dim= fc_dim_in,out_dim=2, hidden_list= [256, 256, 256])
        self.fc1                    = nn.Linear(fc_dim_in,hidden_dim)
        self.fc2                    = nn.Linear(hidden_dim, num_classes)

        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)
    
    @staticmethod
    def make_coord(cls, shape, ranges=None, flatten=True):
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret
    
    




    def forward(self, coarse_features, features,point_coords,coarse_sem_seg_logits):
        # print("fine_grained_features", fine_grained_features.shape)
        # print("coarse_feature",coarse_features.shape)
        feat_shape = features['p2'].size()
        inp_shape = feat_shape
        # print("inp_shape",inp_shape)
        # print("inp shape",inp_shape.shape)
        self.feat_coord = self.make_coord(self,inp_shape[-2:], flatten=False).cuda() \
                            .permute(2,0,1)\
                            .unsqueeze(0).expand(inp_shape[0], 2, *inp_shape[-2:])
        
        

        self.feat = features["p2"]
        # print("features[p1]e")
        # print(features["p2"].shape)
        
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        # StandardPointHead.gen_feat(features['p2'])
        feat = self.feat
        coef = self.coeff
        freq = self.freqq
        # H = self.input_height
        # W = self.input_width
        # print("H",H)
        # print("W",W)
        # print("coarse_sem_seg_logits shape",coarse_sem_seg_logits.shape)
        cell = torch.ones_like(point_coords)
        # print("cell shape",cell.shape)
        # print("cell[:,0] shape",cell[:,0])
        # print("(coarse_sem_seg_logits.shape[-2])",coarse_sem_seg_logits.shape[-2])
        # print("(coarse_sem_seg_logits.shape[-2]*2)",coarse_sem_seg_logits.shape[-2]*2)
        
        cell[:, 0] *= 2 / (coarse_sem_seg_logits.shape[-2])
        cell[:, 1] *= 2 / (coarse_sem_seg_logits.shape[-1])

     

        # print("FC",coef.shape, freq.shape)
        # torch.save(freq,"/home/user/text_inr/detectron2/projects/PointRend/lr_images/analysis/freq/freq2.pt")

        # torch.save(coef,"/home/user/text_inr/detectron2/projects/PointRend/lr_images/analysis/coef/coef2.pt")
        

        """
        # feature map
        print("x shape",feat.shape)
        print("coarse sem seg logit shape",feat.shape)
        activation_map_coarse = feat.squeeze(0)
        activation_map_coarse = activation_map_coarse.mean(dim=0)
        min_value2 = torch.min(activation_map_coarse)
        max_value2 = torch.max(activation_map_coarse)
        normalized2 = (activation_map_coarse - min_value2) / (max_value2 - min_value2)
        normalized2 = np.array(normalized2.cpu())*255
        pro_img2 = Image.fromarray(normalized2.astype(np.uint8)).convert("P")
        pro_img2.save("/home/user/text_inr/detectron2/projects/PointRend/lr_images/feature_map/coarse_feature_c01133.png")
        """
        
        # residual map
        """
        print("coef shape",coef.shape)
        print("freq shape",freq.shape)
        activation_map = freq.squeeze(0)
        activation_map = activation_map
        print("activation map shape",activation_map.shape)
        min_value = torch.min(activation_map)
        max_value = torch.max(activation_map)
        normalized = (activation_map - min_value) / (max_value - min_value)
        normalized = np.array(normalized.cpu())*255
        print("normalized",normalized.shape)
        np.save("/home/user/text_inr/detectron2/projects/PointRend/lr_images/analysis/np_freq/freq3.npy",normalized)
        """
        # pro_img = Image.fromarray(normalized.astype(np.uint8)).convert("P")
        # pro_img.save("/home/user/text_inr/detectron2/projects/PointRend/lr_images/feature_map/residualmap_c01133.png")
        

        

        vx_lst = [-1,1]
        vy_lst = [-1,1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = point_coords.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = point_coords - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2] # 원래의 좌표값만큼 곱해짐
                rel_coord[:, :, 1] *= feat.shape[-1] # 

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = point_coords.shape[:2]

                # print("coords",point_coords.shape, q_coef.shape, q_freq.shape, q_coord.shape)
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)


                # print("q_freq feature map",q_freq.shape)
            
                x = torch.mul(q_coef, q_freq)
                # print("x_featuremap")
                # x_reshape = x.squeeze(0).reshape(45,45,-1)
                # print("x_reshape",x_reshape.shape)
                
                
                # print("torch.mul shape",x.shape)
                # print("coarse_features", coarse_features.shape, x.shape)
                # x = torch.cat((x,coarse_features.permute(0,2,1)), dim=2)
                """
                # print("x shape",x.shape)
                fft_result = torch.fft.fft(x, dim=1)

                # 주파수 데이터의 크기 계산
                magnitude = torch.abs(fft_result)

                # 고주파 성분 추출 및 시각화 (예: 전체 주파수 중 75% 이상)
                full_freq_magnitude = magnitude[0, :, 0].cpu().numpy() 

                # 고주파 성분의 평균 크기 계산 (첫 번째 feature)
                # high_freq_magnitude = high_freq_part[0, :, 0].cpu().numpy()

                # 시각화
                plt.figure(figsize=(10, 4))
                plt.plot(full_freq_magnitude)
                plt.title('Frequency Spectrum')
                plt.xlabel('Frequency Index')
                plt.ylabel('Magnitude')

                # 파일로 저장
                plt.savefig("/home/user/text_inr/pointrend2/pointrend/projects/PointRend/observe/frequency/x.png")
                """

                pred = self.imnet(x.contiguous().view(bs * q, -1)).view(bs, q, -1)
            
            
                    
                
                preds.append(pred)
                
                area = torch.abs(rel_coord[:,:,0]*rel_coord[:,:,1])
                areas.append(area + 1e-9)

        preds = preds
        areas = areas
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        """
        print("f grid sampe",F.grid_sample(coarse_sem_seg_logits,point_coords.flip(-1).unsqueeze(1), mode='bilinear', \
                             padding_mode = 'border', align_corners=False)[:,:,0,:] \
                             .permute(0, 2, 1).shape)
        
        lr_image = F.interpolate(input = coarse_sem_seg_logits, scale_factor=4, mode="nearest")
        print("lr shape1",lr_image.shape)
        print("lr",lr_image)
        
        ub_lr_image = torch.argmax(lr_image[0],dim=0)
        print("lr shape2",lr_image.shape)
        print("lr2",lr_image)

        label = np.array(ub_lr_image.cpu())
        label = np.where(label==1,255,label)
        print("label",label.shape)
        png = Image.fromarray(label.astype(np.uint8)).convert("P")
        png.save("/home/user/text_inr/detectron2/projects/PointRend/lr_images/lr1.png")
        """

        # print("ret shape",ret.shape)
        # label = np.array
        # print("ret shape",ret.shape)
        # print("coarse features shape",coarse_features.shape)
        
        ret += coarse_features.permute(0,2,1)
        
        
        ret = ret.permute(0,2,1)
        # print("ret shape",ret.shape)
        # print("ret",ret)
        
        # print("ret",ret)

        
        
        return ret


@POINT_HEAD_REGISTRY.register()
class ImplicitPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained features and instance-wise MLP parameters as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            channels: the output dimension of each FC layers
            num_layers: the number of FC layers (including the final prediction layer)
            image_feature_enabled: if True, fine-grained image-level features are used
            positional_encoding_enabled: if True, positional encoding is used
        """
        super(ImplicitPointHead, self).__init__()
        # fmt: off
        self.num_layers                         = cfg.MODEL.POINT_HEAD.NUM_FC + 1
        self.channels                           = cfg.MODEL.POINT_HEAD.FC_DIM
        self.image_feature_enabled              = cfg.MODEL.IMPLICIT_POINTREND.IMAGE_FEATURE_ENABLED
        self.positional_encoding_enabled        = cfg.MODEL.IMPLICIT_POINTREND.POS_ENC_ENABLED
        self.num_classes                        = (
            cfg.MODEL.POINT_HEAD.NUM_CLASSES if not cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK else 1
        )
        self.in_channels                        = input_shape.channels
        # fmt: on

        if not self.image_feature_enabled:
            self.in_channels = 0
        if self.positional_encoding_enabled:
            self.in_channels += 256
            self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, 128)))

        assert self.in_channels > 0

        num_weight_params, num_bias_params = [], []
        assert self.num_layers >= 2
        for l in range(self.num_layers):
            if l == 0:
                # input layer
                num_weight_params.append(self.in_channels * self.channels)
                num_bias_params.append(self.channels)
            elif l == self.num_layers - 1:
                # output layer
                num_weight_params.append(self.channels * self.num_classes)
                num_bias_params.append(self.num_classes)
            else:
                # intermediate layer
                num_weight_params.append(self.channels * self.channels)
                num_bias_params.append(self.channels)

        self.num_weight_params = num_weight_params
        self.num_bias_params = num_bias_params
        self.num_params = sum(num_weight_params) + sum(num_bias_params)

    def forward(self, fine_grained_features, point_coords, parameters):
        # features: [R, channels, K]
        # point_coords: [R, K, 2]
        num_instances = fine_grained_features.size(0)
        num_points = fine_grained_features.size(2)

        if num_instances == 0:
            return torch.zeros((0, 1, num_points), device=fine_grained_features.device)

        if self.positional_encoding_enabled:
            # locations: [R*K, 2]
            locations = 2 * point_coords.reshape(num_instances * num_points, 2) - 1
            locations = locations @ self.positional_encoding_gaussian_matrix.to(locations.device)
            locations = 2 * np.pi * locations
            locations = torch.cat([torch.sin(locations), torch.cos(locations)], dim=1)
            # locations: [R, C, K]
            locations = locations.reshape(num_instances, num_points, 256).permute(0, 2, 1)
            if not self.image_feature_enabled:
                fine_grained_features = locations
            else:
                fine_grained_features = torch.cat([locations, fine_grained_features], dim=1)

        # features [R, C, K]
        mask_feat = fine_grained_features.reshape(num_instances, self.in_channels, num_points)

        weights, biases = self._parse_params(
            parameters,
            self.in_channels,
            self.channels,
            self.num_classes,
            self.num_weight_params,
            self.num_bias_params,
        )

        point_logits = self._dynamic_mlp(mask_feat, weights, biases, num_instances)
        point_logits = point_logits.reshape(-1, self.num_classes, num_points)

        return point_logits

    @staticmethod
    def _dynamic_mlp(features, weights, biases, num_instances):
        assert features.dim() == 3, features.dim()
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.einsum("nck,ndc->ndk", x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _parse_params(
        pred_params,
        in_channels,
        channels,
        num_classes,
        num_weight_params,
        num_bias_params,
    ):
        assert pred_params.dim() == 2
        assert len(num_weight_params) == len(num_bias_params)
        assert pred_params.size(1) == sum(num_weight_params) + sum(num_bias_params)

        num_instances = pred_params.size(0)
        num_layers = len(num_weight_params)

        params_splits = list(
            torch.split_with_sizes(pred_params, num_weight_params + num_bias_params, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                # input layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, in_channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            elif l < num_layers - 1:
                # intermediate layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, channels, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, channels, 1)
            else:
                # output layer
                weight_splits[l] = weight_splits[l].reshape(num_instances, num_classes, channels)
                bias_splits[l] = bias_splits[l].reshape(num_instances, num_classes, 1)

        return weight_splits, bias_splits


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)