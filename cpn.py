import os
import json
import uuid

import torch
from torch.optim import AdamW


import cpn_model
import cpn_epoch_cpn
import cpn_epoch_en
from cpn_utils import EpochType
from experiment import experiment
import stim_model


class CPN_EN_CoProc(experiment.CoProc):
    def __init__(self, cfg, retain_stim_grads=False, log_dir=None):
        self.cfg = cfg

        in_dim, stim_dim, out_dim, cuda = cfg.unpack()

        self.cpn = cpn_model.CPNModelLSTM(
            in_dim,
            stim_dim,
            num_neurons=in_dim,
            activation_func=torch.nn.Tanh,
            cuda=cuda,
        )
        self.opt_cpn = AdamW(self.cpn.parameters(), lr=1e-3)

        en_in_dim = cfg.observer_instance.out_dim + stim_dim + 1
        self.en, self.opt_en = stim_model.get_stim_model(
            en_in_dim, out_dim, cuda=cfg.cuda
        )


        self.retain_stim_grads = retain_stim_grads

        self.stims = None

        self.epoch_type = EpochType.EN
        for param in self.cpn.parameters():
            param.requires_grad = False
        for param in self.en.parameters():
            param.requires_grad = True

        self.en_epoch = cpn_epoch_en.CPNEpochEN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg
        )
        self.cpn_epoch = cpn_epoch_cpn.CPNEpochCPN(
            self.en, self.opt_en, self.cpn, self.opt_cpn, self.cfg
        )

        self.uuid = uuid.uuid4()
        if log_dir is not None:
            self.log_dir = os.path.join(log_dir, str(self.uuid))
            os.makedirs(self.log_dir)
        else:
            self.log_dir = None

        self.reset_soft()

    def reset_soft(self):
        self.stims = []
        self.en_epoch.reset()
        self.cpn_epoch.reset()

    def reset(self):
        self.cpn_epoch.reset_period()
        self.epoch_type = EpochType.EN

        self.en, self.opt_en = self.en_epoch.reset_en()
        self.cpn_epoch.set_en(self.en, self.opt_en)

        for param in self.cpn.parameters():
            param.requires_grad = False
        for param in self.en.parameters():
            param.requires_grad = True

        self.reset_soft()

    def forward(self, brain_data, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            new_stim = self.en_epoch.forward(brain_data, loss_history,
                    self.epoch_type == EpochType.VAL_EN)
        elif self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN):
            new_stim = self.cpn_epoch.forward(brain_data, loss_history,
                    self.epoch_type == EpochType.VAL_CPN)
        else:
            # For now: no other thing
            raise ValueError(self.epoch_type)

        if self.retain_stim_grads:
            new_stim.retain_grad()
        self.stims.append(new_stim)

        return new_stim

    def feedback(self, actuals, targets, trial_end, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            self.en_epoch.feedback(
                actuals,
                targets,
                trial_end,
                loss_history,
                is_validation=self.epoch_type == EpochType.VAL_EN,
            )
            update_task_loss = False
        elif self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN):
            self.cpn_epoch.feedback(
                actuals,
                targets,
                trial_end,
                loss_history,
                is_validation=self.epoch_type == EpochType.VAL_CPN,
            )
            update_task_loss = True

        return update_task_loss

    def finish(self, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            (
                self.en,
                self.opt_en,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.en_epoch.finish(loss_history,
                    self.epoch_type == EpochType.VAL_EN)

            if en_is_ready:
                self.epoch_type = EpochType.CPN
                self.cpn_epoch.set_en(self.en, self.opt_en)

                for param in self.cpn.parameters():
                    param.requires_grad = True
                for param in self.en.parameters():
                    param.requires_grad = False

            elif next_is_validation:
                self.epoch_type = EpochType.VAL_EN
            else:
                self.epoch_type = EpochType.EN

            we_are_done = False

        else:
            (
                we_are_done,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.cpn_epoch.finish(loss_history,
                    self.epoch_type == EpochType.VAL_EN)

            if en_is_ready:
                if next_is_validation:
                    self.epoch_type = EpochType.VAL_CPN
                else:
                    self.epoch_type = EpochType.CPN
            else:
                for param in self.cpn.parameters():
                    param.requires_grad = False

                self.en, self.opt_en = self.en_epoch.reset_en()

                self.epoch_type = EpochType.EN
                next_is_validation = False

        result = experiment.EpochResult(
            we_are_done,
            next_is_validation,
            user_data,
        )

        self.reset_soft()
        return result

    def report(self, loss_history):
        if self.log_dir is not None:
            if (loss_history.eidx % 50) == 0:
                log_path = os.path.join(self.log_dir, "log")

                with open(log_path, "w") as f:
                    json.dump(loss_history.render(), f)
                    f.flush()

                cpn_path = os.path.join(self.log_dir, "cpn.model")
                en_path = os.path.join(self.log_dir, "en.model")

                torch.save(self.cpn.state_dict(), cpn_path)
                torch.save(self.en.state_dict(), en_path)