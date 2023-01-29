# Copy from Scribble_Saliency
import torch
import torch.nn.functional as F

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx

def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

class SmoothnessLoss(torch.nn.Module):
    def __init__(self, with_scribble=False, alpha = 10.0):
        super(SmoothnessLoss, self).__init__()
        self.with_scribble = with_scribble
        self.alpha = alpha

    def forward(self, pred, target, label, weight=None):
        """
        args:
            pred (Tensor) - [B * clip_length, K, im_h, im_w]
            target (Tensor) - [B * clip_length, im_h, im_w] gray image
            label (Tensor) - [B * clip_length, 384, 384]
            weight (Tensor) - [B * clip_length]
        """
    
        target = target.unsqueeze(1)
        loss = 0

        label = label.bool().unsqueeze(1)

        for o in range(pred.shape[1]):
            loss += self.get_saliency_smoothness(pred[:, o].unsqueeze(1), target, label)
        if weight is not None:
            return (loss * weight).sum() / (weight.sum() + 1e-8)
        else:
            return loss.mean()



    def get_saliency_smoothness(self, pred, gt, label):
        # [B,1,384,384] [B,1,384,384] [B,1,384,384]
        s1 = 10
        s2 = 1
        ## first oder derivative: sobel
        sal_x = torch.abs(gradient_x(pred))
        sal_y = torch.abs(gradient_y(pred))
        gt_x = gradient_x(gt)
        gt_y = gradient_y(gt)
        w_x = torch.exp(torch.abs(gt_x) * (-self.alpha))
        w_y = torch.exp(torch.abs(gt_y) * (-self.alpha))

        if self.with_scribble:
            w_x[label==True] = 1
            w_y[label==True] = 1

        cps_x = charbonnier_penalty(sal_x * w_x)
        cps_y = charbonnier_penalty(sal_y * w_y)
        cps_xy = cps_x + cps_y

        ## second order derivative: laplacian
        lap_sal = torch.abs(laplacian_edge(pred))
        lap_gt = torch.abs(laplacian_edge(gt))
        weight_lap = torch.exp(lap_gt * (-self.alpha))

        if self.with_scribble:
            weight_lap[label==True] = 1

        weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

        smooth_loss = s1 * torch.mean(cps_xy, dim=(1,2,3)) + s2 * torch.mean(weighted_lap, dim=(1,2,3))

        return smooth_loss