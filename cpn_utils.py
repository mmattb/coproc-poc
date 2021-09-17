import enum

import attr
import torch

from experiment import stats
from experiment.utils import render_none_or_float


class EpochType(enum.Enum):
    EN = 0
    CPN = 1
    CPN_AND_EN = 2
    VAL_EN = 3
    VAL_CPN = 4


@attr.s(auto_attribs=True)
class CPNENStats(stats.UserData):
    epoch_type: EpochType
    train_loss: float
    train_val_loss: float
    pred_loss: float
    pred_val_loss: float

    def render(self):
        return {
            "epoch_type": self.epoch_type.name,
            "train_loss": render_none_or_float(self.train_loss),
            "pred_loss": render_none_or_float(self.pred_loss),
            "pred_val_loss": render_none_or_float(self.pred_val_loss),
        }


_G_MSELOSS = torch.nn.MSELoss()


def calc_pred_loss(preds, actuals):
    return _G_MSELOSS(preds, actuals)


def calc_train_loss(preds, targets):
    return _G_MSELOSS(preds, targets[:, 1:, :])
