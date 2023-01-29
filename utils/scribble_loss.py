import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss

class ScribbleLoss(nn.Module):
    def __init__(self, use_focal_loss=True, alpha=0.25):
        super(ScribbleLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha

    def forward(self, binary_scribble_logit, scribble_label, valid_weights=None):
        binary_scribble_logit = binary_scribble_logit[valid_weights == 1]
        scribble_label = scribble_label[valid_weights == 1]
        if self.use_focal_loss:
            scirbble_loss = sigmoid_focal_loss(binary_scribble_logit, scribble_label, 
                                               alpha=self.alpha)
        else:
            raise Exception('Unkown scribble loss!')
        return torch.sum(scirbble_loss) / (torch.sum(valid_weights) + 1e-8)