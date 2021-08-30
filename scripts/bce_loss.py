import torch
import torch.nn as nn


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, targets):
        bce_loss = nn.BCELoss(weight=self.weight)(preds, targets)
        loss = targets * self.alpha * (1. - preds) ** self.gamma * bce_loss + (
                    1. - targets) * preds ** self.gamma * bce_loss
        loss = loss.mean()
        return loss


class criterionLoss(nn.Module):
    def __init__(self, use_focal=False, weight=None):
        super(criterionLoss, self).__init__()
        self.use_focal = use_focal
        self.weight = weight

    def forward(self, output, label):
        output = output.squeeze()
        if self.use_focal:
            loss_fn = BCEFocalLoss(weight=self.weight)
        else:
            loss_fn = nn.BCELoss(weight=self.weight)
        return loss_fn(output, label.float())


def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
