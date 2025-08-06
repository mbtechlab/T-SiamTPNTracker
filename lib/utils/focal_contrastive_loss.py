import torch
import torch.nn.functional as F

def focal_contrastive_loss(logits, targets, alpha=0.5, gamma=1.0, loss_normalizing_factor=None):
    """
    Compute focal loss for logits and labels in PyTorch for binary classification.

    Args:
        logits: [batch_size, 1] float tensor.
        targets: [batch_size, 1] float tensor (values 0 or 1).
        alpha: A float scalar for balancing positive and negative examples.
        gamma: A float scalar for focusing on hard examples.
        loss_normalizing_factor: Optional normalization factor.

    Returns:
        A scalar loss.
    """
    # Remove dimension check for binary classification
    assert alpha <= 1.0 and alpha >= 0

    # Convert targets to float if not already
    targets = targets.float()

    # Compute sigmoid of logits
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=1e-6, max=1 - 1e-6)  # To avoid numerical issues

    # Compute focal loss components
    ce_loss = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    pt = torch.where(targets == 1, probs, 1 - probs)
    modulator = (1 - pt) ** gamma

    loss = modulator * ce_loss
    weighted_loss = torch.where(targets == 1, alpha * loss, (1 - alpha) * loss)

    if loss_normalizing_factor is not None:
        weighted_loss = weighted_loss / (loss_normalizing_factor + 1e-20)

    return weighted_loss.mean()

