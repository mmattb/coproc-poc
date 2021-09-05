import enum
import json

import attr
import torch.nn

import class_stdevs


_G_MSELOSS = torch.nn.MSELoss()

LOSS_RENDER_FMT = "%0.6f"
def _render_none_or_float(val, fmt=LOSS_RENDER_FMT):
    if val is None or val != val:
        return "-"
    else:
        return fmt % val


def calc_pred_loss(actuals, preds, targets):
    return _G_MSELOSS(preds, actuals)


def calc_task_loss(actuals, preds, targets):
    return _G_MSELOSS(actuals, targets[:, 1:, :])


def calc_train_task_loss(actuals, preds, targets):
    return _G_MSELOSS(preds, targets[:, 1:, :])


class LossRecType(enum.Enum):
    EN = 0
    CPN = 1
    CPN_AND_EN = 2


@attr.s(auto_attribs=True)
class LossRec:
    eidx: int
    type: LossRecType
    pred_loss: float
    task_loss: float
    train_loss: float
    pred_val_loss: float
    task_val_loss: float
    pct_recov: float
    class_separation: float


class LossHistory:
    def __init__(self, lesioned_loss, healthy_loss, data_set, var_report_cadence=10):
        self._recs = []

        self.lesioned_loss = lesioned_loss
        self.healthy_loss = healthy_loss

        self.eidx = 0

        self._last_pred_loss = None
        self._last_task_loss = None
        self._last_train_loss = None

        self._last_pred_val_loss = None
        self._last_task_val_loss = None

        var_whole, var_within = \
                class_stdevs.calc_raw_dset_class_vars(data_set)
        self._healthy_class_separation = var_whole / var_within

        self._var_report_cadence = var_report_cadence

        self._max_pct_recov = 0

    def calc_pct_recov(self, task_loss):
        dh = task_loss.item() - self.healthy_loss
        dl = self.lesioned_loss - self.healthy_loss
        recov_pct = 1.0 - (dh / dl)
        return recov_pct

    def calc_class_vars(self, actuals, labels):
        var_whole, var_within = class_stdevs.calc_class_vars(actuals, labels)
        return var_whole, var_within

    def render_rec(self, rec):
        rec_rendered = {
            "eidx": rec.eidx,
            "type": rec.type.name,
            "pred_loss": _render_none_or_float(rec.pred_loss),
            "task_loss": _render_none_or_float(rec.task_loss),
            "train_loss": _render_none_or_float(rec.train_loss),
            "pred_val_loss": _render_none_or_float(rec.pred_val_loss),
            "task_val_loss": _render_none_or_float(rec.task_val_loss),
            "pct_recov": _render_none_or_float(rec.pct_recov, fmt="%0.3f"),
            "class_separation": _render_none_or_float(rec.class_separation, fmt="%0.3f"),
        }
        return rec_rendered

    def render(self):
        rendered = []

        for rec in self._recs:
            rec_rendered = self.render_rec(rec)
            rendered.append(rec_rendered)

        return rendered

    def dump(self):
        return json.dumps(self.render())

    def dump_to_file(self, fname):
        with open(fname, "w") as f:
            json.dump(self.render(), f)

    def report(
        self,
        epoch_type,
        pred_loss,
        task_loss,
        train_loss,
        pred_val_loss=None,
        task_val_loss=None,
        class_separation=None
    ):

        if pred_val_loss is None:
            if len(self._recs) > 0:
                pred_val_loss = self._recs[-1].pred_val_loss
            else:
                pred_val_loss = float("nan")
        else:
            pred_val_loss = pred_val_loss.item()

        if task_val_loss is None:
            if len(self._recs) > 0:
                task_val_loss = self._recs[-1].task_val_loss
            else:
                task_val_loss = float("nan")
        else:
            task_val_loss = task_val_loss.item()

        if class_separation is None:
            if len(self._recs) > 0:
                class_separation = self._recs[-1].class_separation
            else:
                class_separation = float("nan")


        self._last_train_loss = train_loss


        rec = LossRec(
            self.eidx,
            epoch_type,
            None,
            None,
            train_loss.item(),
            pred_val_loss,
            task_val_loss,
            None,
            class_separation,
        )

        rec.pred_loss = pred_loss.item()
        self._last_pred_loss = pred_loss

        if epoch_type in (LossRecType.CPN, LossRecType.CPN_AND_EN):
            rec.task_loss = task_loss.item()
            self._last_task_loss = task_loss
            pct_recov = self.calc_pct_recov(task_loss)
            if pct_recov > self._max_pct_recov:
                self._max_pct_recov = pct_recov
            rec.pct_recov = pct_recov
        elif len(self._recs) > 0:
            rec.task_loss = self._recs[-1].task_loss
            rec.pct_recov = self._recs[-1].pct_recov


        self._recs.append(rec)
        self.eidx += 1

    def report_by_result(self, epoch_type, actuals, preds, dout, labels):
        pred_loss = calc_pred_loss(actuals, preds, dout)
        class_separation = None
        if epoch_type in (LossRecType.CPN, LossRecType.CPN_AND_EN):
            task_loss = calc_task_loss(actuals, preds, dout)
            if (self.eidx % self._var_report_cadence) == 0:
                var_whole, var_within = self.calc_class_vars(actuals, labels)
                class_separation = var_whole / var_within
                class_separation -= self._healthy_class_separation
        else:
            task_loss = None

        train_loss = calc_train_task_loss(actuals, preds, dout)


        self.report(epoch_type, pred_loss, task_loss, train_loss, class_separation=class_separation)

    def report_val_last_result(self, actuals, preds, dout):
        if not self._recs:
            raise ValueError(
                "Can only accumulate validation results on an existing loss"
            )

        prev_rec = self._recs[-1]

        if prev_rec.type in (LossRecType.EN, LossRecType.CPN_AND_EN):
            pred_val_loss = calc_pred_loss(actuals, preds, dout)
            self._last_pred_val_loss = pred_val_loss
            prev_rec.pred_val_loss = pred_val_loss.item()


        if prev_rec.type in (LossRecType.CPN, LossRecType.CPN_AND_EN):
            task_val_loss = calc_task_loss(actuals, preds, dout)
            self._last_task_val_loss = task_val_loss
            self._recs[-1].task_val_loss = task_val_loss.item()

    def get_recent(self):
        if self.eidx != 0:
            return self._recs[-1]
        return None

    @property
    def recent_pred_loss(self):
        return self._last_pred_loss

    @property
    def recent_task_loss(self):
        return self._last_task_loss

    @property
    def recent_train_loss(self):
        return self._last_train_loss

    @property
    def recent_pred_val_loss(self):
        return self._last_pred_val_loss

    @property
    def recent_task_val_loss(self):
        return self._last_task_val_loss

    @property
    def max_pct_recov(self):
        return self._max_pct_recov

    def log(self, logger, msg=None):
        recent = self.get_recent()

        if recent is not None:
            rendered = self.render_rec(recent)
            del rendered["train_loss"]
            del rendered["task_val_loss"]
            del rendered["type"]

            eidx = rendered["eidx"]
            del rendered["eidx"]

            out = f"{eidx} " + " ".join([f"{k}: {v}" for k, v in rendered.items()])

            if msg:
                out = msg + " " + out

            logger.info(out)
