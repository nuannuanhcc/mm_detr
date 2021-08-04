import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_reid_loss_evaluator
import copy

class REIDModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(REIDModule, self).__init__()
        self.cfg = cfg
        self.num = 7
        loss_evaluator = make_reid_loss_evaluator(cfg)
        self.loss_evaluator = nn.ModuleList([copy.deepcopy(loss_evaluator) for _ in range(self.num)])

        fc = nn.Linear(256, 2048)
        self.share_para = False
        if self.share_para:
            self.fc = nn.ModuleList([fc for _ in range(self.num)])
        else:
            self.fc = nn.ModuleList([copy.deepcopy(fc) for _ in range(self.num)])

    def forward(self, x, gt_labels=None):

        if self.training:
            loss_reid = [l(F.normalize(f(i), dim=-1), gt_labels) for i, f, l in zip(x, self.fc, self.loss_evaluator)]
            return {"loss_reid": [torch.mean(torch.stack(loss_reid))], }
        else:
            feats = [F.normalize(f(i), dim=-1) for i, f in zip(x, self.fc)]
            return feats[-2]


def build_reid(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return REIDModule(cfg)
