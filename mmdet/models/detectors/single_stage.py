import warnings

import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from .base import BaseDetector
from mmdet.models.reid_heads.reid import build_reid


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        if test_cfg.with_reid:
            self.reid_head = build_reid(test_cfg)
            self.bbox_roi_extractor = build_roi_extractor(test_cfg.roi_extractor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        cls_labels = [i[:, 0] for i in gt_labels] if self.train_cfg.with_reid else gt_labels
        losses_all = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              cls_labels, gt_bboxes_ignore)
        losses = losses_all[0]
        de_bboxes = losses_all[1]

        # detection
        if not self.train_cfg.with_reid:
            return losses

        # person search
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], bbox2roi(de_bboxes[-1]))
        loss_reid = self.reid_head(bbox_feats, gt_labels)
        losses.update(loss_reid)
        return losses

    def simple_test(self, img, img_metas, rescale=False, gt_bboxes=None):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # person search -- query
        if gt_bboxes is not None:
            feat = self.extract_feat(img)
            results_list = self.bbox_head.simple_test(
                feat, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            pre_bbox_list = results_list[0][0] * img_metas[0]['scale_factor'][0]
            index = self.bbox_iou(pre_bbox_list.cpu().numpy(), gt_bboxes[0][0][0].cpu().numpy())
            pre_bbox_feats = self.bbox_roi_extractor(
                feat[:self.bbox_roi_extractor.num_inputs], bbox2roi([pre_bbox_list[index:index + 1]]))
            pre_bbox_feats = self.reid_head(pre_bbox_feats)
            return [[bbox_results[0][0][index:index + 1]]], pre_bbox_feats.cpu().numpy()

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        # only detection
        if not self.test_cfg.with_reid:
            return bbox_results
        
        # person search -- gallery
        pre_bbox_list = results_list[0][0] * img_metas[0]['scale_factor'][0]
        pre_bbox_feats = self.bbox_roi_extractor(
            feat[:self.bbox_roi_extractor.num_inputs], bbox2roi([pre_bbox_list]))
        pre_bbox_feats = self.reid_head(pre_bbox_feats)
        return bbox_results, pre_bbox_feats.cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels

    def bbox_iou(self, boxes, gt):
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            areas_boxes = (x2 - x1 + 1) * (y2 - y1 + 1)
            areas_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])

            x1_max = np.maximum(gt[0], boxes[:, 0])
            y1_max = np.maximum(gt[1], boxes[:, 1])
            x2_min = np.minimum(gt[2], boxes[:, 2])
            y2_min = np.minimum(gt[3], boxes[:, 3])

            w = np.maximum(x2_min - x1_max + 1, 0.0)
            h = np.maximum(y2_min - y1_max + 1, 0.0)
            inter = w * h
            union = areas_boxes + areas_gt - inter
            iou = inter / union
            if iou.max() < 0.8:
                print("###############", iou.max())
            return iou.argmax()