from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
import numpy as np

class SiamTPNSTActor(BaseActor):
    """ Actor for training the SIAMTPN_ST (Stage 1)"""
    def __init__(self, net, objective, loss_weight, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.cfg = cfg
        # Generate anchor grids based on configuration parameters
        self.grids = self._generate_anchors(cfg.MODEL.ANCHOR.NUM, cfg.MODEL.ANCHOR.FACTOR, cfg.MODEL.ANCHOR.BIAS)

    def _generate_anchors(self, num=20, factor=1, bias=0.5):
        """
        Generate anchors for each sampled point.
        Args:
            num: Number of points per dimension.
            factor: Scaling factor for the anchors.
            bias: Bias added to the anchors.
        Returns:
            grids: A tensor containing anchor coordinates.
        """
        x = np.arange(num)
        y = np.arange(num)
        xx, yy = np.meshgrid(x, y) 
        xx = (factor * xx + bias) / num
        yy = (factor * yy + bias) / num
        xx = torch.from_numpy(xx).view(-1).float()
        yy = torch.from_numpy(yy).view(-1).float()
        grids = torch.stack([xx, yy], -1)  # Shape: N x 2
        return grids

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'template', 'search', 'gt_bbox'.
                - template_images: (N_t, batch, 3, H, W)
                - search_images: (N_s, batch, 3, H, W)
        Returns:
            loss: The computed training loss.
            status: A dictionary containing detailed loss metrics.
        """
        # Perform forward pass
        output = self.forward_pass(data, run_box_head=True, run_cls_head=False) 

        target = {}
        target['anno'] = data['search_anno'][0]  # Ground-truth bounding box (batch, 4) (x1, y1, w, h)
        target['label'] = data['search_label'][0]  # Ground-truth labels (B, 400)
        target['centerness'] = data['centerness'][0]  # Centerness target

        # Compute losses
        loss, status = self.compute_losses(output, target)

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        """
        Perform a forward pass through the model.
        Args:
            data: Input data containing template and search images.
            run_box_head: Whether to run the bounding box head.
            run_cls_head: Whether to run the classification head.
        Returns:
            output: Model predictions.
        """
        # Process the template images
        feat_dict_list = []
        for i in range(self.cfg.DATA.TEMPLATE.NUMBER):
            template_img_i = data['template_images'][i]  # Template image (batch, 3, 128, 128)
            feat_dict_list.append(template_img_i)

        train_imgs = feat_dict_list  # List of template images
        test_img = data['search_images'][0]  # Search image

        # Call the model with template and search images
        output = self.net(train_imgs, test_img, run_cls_head=run_cls_head, run_box_head=run_box_head)
        
        return output  # Output shape: (B, N, 4), (B, N, 1)

    def compute_losses(self, output, target):
        """
        Compute losses for the predictions.
        Args:
            output: Model predictions.
            target: Ground-truth values for labels and bounding boxes.
        Returns:
            loss: Total loss.
            status: A dictionary with detailed loss metrics.
        """
        # Extract predicted classes and boxes
        first_key = list(output.keys())[0]
        pred_class = output[first_key]  # Predicted class scores, shape: [B * N, 1]
        second_key = list(output.keys())[1]
        pred_boxes = output[second_key]  # Predicted boxes, shape: [B, N, 4]
        
        # Get ground-truth class labels
        gt_class = target['label'].view(pred_class.size(0), -1).float() 
        # Compute classification loss
        cls_loss = self.objective['ce'](pred_class, gt_class)  # For BCEWithLogitsLoss

        # Compute centerness mask
        gt_centerness = target['centerness'].view(-1)
        mask = (gt_centerness > 0).to(pred_class.dtype)

        # Process predicted boxes
        B, N, _ = pred_boxes.shape
        grids = self.grids[None, ...].repeat(B, 1, 1).to(pred_boxes.device)
        lt = grids[:, :, :2] - pred_boxes[:, :, :2]  # Left-top coordinates
        rb = grids[:, :, :2] + pred_boxes[:, :, 2:]  # Right-bottom coordinates
        pred_boxes = torch.cat([lt, rb], -1).view(-1, 4)

        # Process ground-truth boxes
        gt_boxes = box_xywh_to_xyxy(target['anno']).clamp(min=0.0, max=1.0)  # Convert to (x1, y1, x2, y2)
        gt_boxes = gt_boxes[:, None, :].repeat(1, N, 1).view(-1, 4)

        # Compute GIoU and IoU losses
        giou_loss, iou = self.objective['giou'](pred_boxes, gt_boxes)  # Generalized IoU loss
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes, gt_boxes, reduction='none')  # L1 loss

        # Apply mask and compute final losses
        giou_loss = (giou_loss * mask).sum() / mask.sum()
        l1_loss = (l1_loss.sum(1) * mask).sum() / mask.sum()
        mean_iou = (iou.detach() * mask).sum() / mask.sum()
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss 

        loss = loss + self.loss_weight['ce'] * cls_loss
        # Prepare status dictionary
        status = {
            "Loss/total": loss.item(),
            "Loss/giou": giou_loss.item(),
            "Loss/l1": l1_loss.item(),
            "IoU": mean_iou.item()
        }
        status["Loss/cls"] = cls_loss.item()
        return loss, status
