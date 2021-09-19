import torch

from cpn_utils import CPNENStats, calc_pred_loss, calc_train_loss, EpochType
import experiment.utils as utils


class CPNEpochCPN:
    def __init__(self, en, opt_en, cpn, opt_cpn, cfg):
        self.en = en
        self.opt_en = opt_en

        self.cpn = cpn
        self.opt_cpn = opt_cpn
        for p in opt_cpn.param_groups:
            p["lr"] = 1e-3

        self.cfg = cfg

        self.preds = None
        self.tidx = 0

        # This tracks how many epochs we've been training the CPN
        self.checkpoint_eidx = 0

        self.recent_task_losses = []

        self.recent_train_loss = 0.05
        self.recent_pred_loss = 0.05
        self.recent_train_val_loss = 0.05

        # Weight for stim regularizer
        self.reg_stim_weight = 3e-7
        self.stims = []

        self.reset()

    def reset(self):
        self.preds = None
        self.tidx = 0
        self.stims = []
        self.reset_models()

    def reset_models(self):
        self.en.reset()
        self.cpn.reset()
        self.opt_en.zero_grad()
        self.opt_cpn.zero_grad()

    def reset_period(self):
        self.checkpoint_eidx = 0
        self.recent_task_losses = []

    def set_en(self, en, opt_en):
        self.en = en
        self.opt_en = opt_en
        self.en.reset()
        self.opt_en.zero_grad()

    def ensure_preds(self, batch_size):
        if self.preds is None:
            self.preds = []

    def forward(self, brain_data, loss_history, is_validation):
        batch_size = brain_data[0].shape[0]
        self.ensure_preds(batch_size)

        cpn_in = torch.cat(brain_data, axis=1)

        new_stim = self.cpn(cpn_in)

        # en receives (obs, stims, trial_end)
        new_obs_en = brain_data[0]
        en_in = torch.cat((new_obs_en, new_stim, brain_data[-1]), axis=1)
        cur_pred = self.en(en_in)

        self.preds.append(cur_pred.unsqueeze(dim=1))

        self.tidx += 1
        self.stims.append(new_stim)

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history, is_validation):
        preds = torch.cat(self.preds, axis=1)
        preds = utils.trunc_to_trial_end(preds, trial_end[:, :-1, :])

        pred_loss = calc_pred_loss(preds, actuals)
        train_loss = calc_train_loss(preds, targets)

        # Regularization for stimulation applied
        train_loss += self.reg_stim_weight * sum(
            [torch.linalg.norm(s) for s in self.stims]
        )

        if is_validation:
            self.recent_train_val_loss = train_loss.item()
        else:
            self.recent_train_loss = train_loss.item()
            train_loss.backward(inputs=list(self.cpn.parameters()))

        self.recent_pred_loss = pred_loss.item()

    def finish(self, loss_history, is_validation):

        rtl = self.recent_train_loss
        if rtl is None or rtl >= 0.008:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-3
        elif rtl >= 0.006:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 5e-5
        elif rtl >= 0.002:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 2e-6
        else:
            for p in self.opt_cpn.param_groups:
                p["lr"] = 1e-6

        # Every 10 epochs let's validate/test
        if not is_validation and (loss_history.eidx % 10) == 0:
            next_is_validation = True
        else:
            next_is_validation = False

        last_rec = loss_history.get_recent_record(-2)
        pred_val_loss_out = float("nan")
        if last_rec is not None:
            last_user_data = last_rec.user_data
            if last_user_data is not None:
                pred_val_loss_out = last_user_data.pred_val_loss

        rtl = loss_history.recent_task_loss
        self.recent_task_losses.append(rtl)
        if loss_history.max_pct_recov > 0.9:
            we_are_done = True
            en_is_ready = False
        else:
            we_are_done = False

            if self.recent_pred_loss > max(rtl / 10, 6e-4):
                en_is_ready = False
                self.reset_period()
            elif self.checkpoint_eidx >= 100:
                en_is_ready = False
                self.reset_period()
            elif self.checkpoint_eidx > 30:
                num_reg = 0
                for l in self.recent_task_losses[-30:]:
                    if l < rtl:
                        num_reg += 1

                if num_reg > 15:
                    en_is_ready = False
                    self.reset_period()
                else:
                    en_is_ready = True
                    self.opt_cpn.step()
                    self.checkpoint_eidx += 1
            else:
                en_is_ready = True
                self.opt_cpn.step()
                self.checkpoint_eidx += 1

        user_data = CPNENStats(
            "cpn",
            EpochType.CPN,
            self.recent_train_loss,
            self.recent_train_val_loss,
            self.recent_pred_loss,
            pred_val_loss_out,
        )

        self.reset()
        return we_are_done, next_is_validation, en_is_ready, user_data
