import torch
from torchvision.ops.boxes import box_area
import numpy as np
import math

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# Modified from torchvision to also return the union
# Note that this function only supports shape (N,4)
def box_iou(boxes1, boxes2):
    """
    Calculate the IoU (Intersection over Union) between two sets of boxes.
    Args:
        boxes1: Tensor of shape (N, 4) (x1, y1, x2, y2)
        boxes2: Tensor of shape (N, 4) (x1, y1, x2, y2)
    Returns:
        iou: Tensor containing IoU values for each pair of boxes.
        union: Tensor containing union areas for each pair of boxes.
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N, 2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N, 2)

    wh = (rb - lt).clamp(min=0)  # (N, 2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union

# Note that this implementation is different from DETR's
def generalized_box_iou(boxes1, boxes2):
    """
    Calculate Generalized IoU as described in https://giou.stanford.edu/.
    Args:
        boxes1: Tensor of shape (N, 4) (x1, y1, x2, y2)
        boxes2: Tensor of shape (N, 4) (x1, y1, x2, y2)
    Returns:
        giou: Generalized IoU values.
        iou: IoU values.
    """
    # Ensure boxes are valid
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)  # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N, 2)
    area = wh[:, 0] * wh[:, 1]  # (N,)

    return iou - (area - union) / area, iou

def giou_loss(boxes1, boxes2):
    """
    Calculate Generalized IoU loss.
    Args:
        boxes1: Tensor of shape (N, 4) (x1, y1, x2, y2)
        boxes2: Tensor of shape (N, 4) (x1, y1, x2, y2)
    Returns:
        loss: GIoU loss values.
        iou: IoU values.
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return 1 - giou, iou

def compute_diou(output, target):
    """
    Calculate Distance IoU (DIoU).
    Args:
        output: Tensor of shape (N, 4), predicted boxes (x1, y1, x2, y2).
        target: Tensor of shape (N, 4), ground truth boxes (x1, y1, x2, y2).
    Returns:
        diouk: DIoU values.
        iouk: IoU values.
    """
    x1, y1, x2, y2 = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    x1g, y1g, x2g, y2g = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u

    return diouk, iouk

def clip_box(box: list, H, W, margin=0):
    """
    Clip the box coordinates to ensure they lie within the image boundaries.
    Args:
        box: List of box coordinates [x1, y1, w, h].
        H: Height of the image.
        W: Width of the image.
        margin: Optional margin to leave around the box.
    Returns:
        Clipped box coordinates as [x1, y1, w, h].
    """
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]

def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> (torch.Tensor, torch.Tensor):
    """
    Complete Intersection over Union (CIoU) Loss.
    Args:
        boxes1: Tensor of box coordinates in XYXY format (N, 4) or (4,).
        boxes2: Tensor of box coordinates in XYXY format (N, 4) or (4,).
        reduction: Specifies the reduction type: 'none' | 'mean' | 'sum'.
        eps: Small number to prevent division by zero.
    Returns:
        loss: Tensor containing CIoU loss values.
        iou: Tensor containing IoU values.
    """
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Ensure x1 <= x2 and y1 <= y2
    assert (x2 >= x1).all(), "Invalid boxes: x1 larger than x2"
    assert (y2 >= y1).all(), "Invalid boxes: y1 larger than y2"

    # Intersection coordinates
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (xkis2 > xkis1) & (ykis2 > ykis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # Smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = (xc2 - xc1).pow(2) + (yc2 - yc1).pow(2) + eps

    # Center points
    x_p = (x1 + x2) / 2
    y_p = (y1 + y2) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = (x_p - x_g).pow(2) + (y_p - y_g).pow(2)

    # Aspect ratio consistency
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w_gt / (h_gt + eps)) - torch.atan(w_pred / (h_pred + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Compute CIoU loss
    loss = 1 - iou + (distance / diag_len) + alpha * v

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        iou = iou.mean()
    elif reduction == "sum":
        loss = loss.sum()
        iou = iou.sum()

    return loss, iou
