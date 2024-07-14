# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F

from detectron2.layers import cat, shapes_to_tensor
from detectron2.structures import BitMasks, Boxes


"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def generate_regular_grid_point_coords(R, side_size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def sample_sdf_map_points(sdf_map, num_points=10000, low=-0.05, high=0.05):
    """
    SDF map에서 주어진 범위 내의 점을 샘플링하고, 필요한 경우 패딩하여 반환합니다.

    Args:
        sdf_map (torch.Tensor): 입력 SDF 맵 (batch, 128, 256 크기).
        num_points (int): 샘플링할 총 점의 수 (기본값: 10000).
        low (float): 샘플링할 값의 최저 범위 (기본값: -0.05).
        high (float): 샘플링할 값의 최고 범위 (기본값: 0.05).

    Returns:
        point_indices (Tensor): 샘플링된 좌표의 인덱스 (batch, num_points).
        point_coords (Tensor): 샘플링된 좌표의 정규화된 값 (batch, num_points, 2).
    """
    batch_size, height, width = sdf_map.shape
    num_points = min(height * width, num_points)
    sampled_indices = torch.full((batch_size, num_points), -1, dtype=torch.int32)
    point_coords = torch.zeros(batch_size, num_points, 2, dtype=torch.float32, device=sdf_map.device)
    
    h_step = 1.0 / float(height)
    w_step = 1.0 / float(width)

    for b in range(batch_size):
        # 범위 내의 점들 선택
        mask = (sdf_map[b] >= low) & (sdf_map[b] <= high)
        valid_points = torch.nonzero(mask, as_tuple=False)

        num_valid_points = valid_points.shape[0]

        if num_valid_points >= num_points:
            # 랜덤하게 선택
            indices = torch.randperm(num_valid_points)[:num_points]
            selected_points = valid_points[indices]
        elif num_valid_points > 0:
            # 모든 유효한 점을 선택하고, 나머지 부분을 패딩
            selected_points = valid_points
            padding_indices = torch.randint(0, num_valid_points, (num_points - num_valid_points,))
            padding_points = valid_points[padding_indices]
            selected_points = torch.cat((valid_points, padding_points), dim=0)
        else:
            # 유효한 점이 없는 경우, 전체 범위에서 랜덤하게 선택
            selected_points = torch.stack((torch.randint(0, width, (num_points,)), torch.randint(0, height, (num_points,))), dim=-1)

        # 인덱스 및 좌표 저장
        selected_indices = selected_points[:, 1] * width + selected_points[:, 0]
        sampled_indices[b, :] = selected_indices

        point_coords[b, :, 0] = w_step / 2.0 + selected_points[:, 0].to(torch.float32) * w_step
        point_coords[b, :, 1] = h_step / 2.0 + selected_points[:, 1].to(torch.float32) * h_step

    return sampled_indices, point_coords



def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def point_sample_fine_grained_features(features_list, feature_scales, boxes, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
            together.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    cat_boxes = Boxes.cat(boxes)
    num_boxes = [b.tensor.size(0) for b in boxes]

    point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image, num_boxes)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = shapes_to_tensor([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale.to(feature_map.device)
            point_features_per_image.append(
                point_sample(
                    feature_map[idx_img].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(1, 0)
            )
        point_features.append(cat(point_features_per_image, dim=1))

    return cat(point_features, dim=0), point_coords_wrt_image


def get_point_coords_wrt_image(boxes_coords, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.

    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image


def sample_point_labels(instances, point_coords):
    """
    Sample point labels from ground truth mask given point_coords.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. So, i_th elememt of the list contains R_i objects and R_1 + ... + R_N is
            equal to R. The ground-truth gt_masks in each instance will be used to compute labels.
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            instances and P is the number of points for each instance. The coordinates are in
            the absolute image pixel coordinate space, i.e. [0, H] x [0, W].

    Returns:
        Tensor: A tensor of shape (R, P) that contains the labels of P sampled points.
    """
    with torch.no_grad():
        gt_mask_logits = []
        point_coords_splits = torch.split(
            point_coords, [len(instances_per_image) for instances_per_image in instances]
        )
        for i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            assert isinstance(
                instances_per_image.gt_masks, BitMasks
            ), "Point head works with GT in 'bitmask' format. Set INPUT.MASK_FORMAT to 'bitmask'."

            gt_bit_masks = instances_per_image.gt_masks.tensor
            h, w = instances_per_image.gt_masks.image_size
            scale = torch.tensor([w, h], dtype=torch.float, device=gt_bit_masks.device)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask_logits.append(
                point_sample(
                    gt_bit_masks.to(torch.float32).unsqueeze(1),
                    points_coord_grid_sample_format,
                    align_corners=False,
                ).squeeze(1)
            )

    point_labels = cat(gt_mask_logits)
    return point_labels
