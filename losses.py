import torch
import torch.nn.functional as F

def dice_loss(y_pred, y_test):
    """Dice Loss function for binary classification.
    
    pred: tensor with first dimension as batch (logits before sigmoid)
    target: tensor with first dimension as batch (binary ground truth)
    
    Returns the Dice Loss, which is differentiable.
    """
    
    smooth = 1.0
    
    pred = torch.sigmoid(y_pred)
    
    if pred.size() != y_test.size():
        pred = F.interpolate(pred, size=y_test.size()[2:], mode='bilinear', align_corners=False)
    
    iflat = pred.contiguous().view(-1)
    tflat = y_test.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = iflat.sum()
    B_sum = tflat.sum()
    dice_score = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return 1 - dice_score
