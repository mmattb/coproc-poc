import logging
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

g_logger = logging.getLogger("cpn")


class CPN_EN_CoProc(experiment.CoProc):
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg

        in_dim, stim_dim, out_dim, cuda = cfg.unpack()

        self.cpn = cpn_model.CPNModelLSTM(
            in_dim,
            stim_dim,
            num_neurons=in_dim,
            activation_func=cfg.cpn_activation,
            cuda=cuda,
        )
        self.opt_cpn = AdamW(self.cpn.parameters(), lr=1e-3)

        en_in_dim = cfg.observer_instance.out_dim + stim_dim + 1
        self.en, self.opt_en = stim_model.get_stim_model(
            en_in_dim, out_dim, activation=cfg.en_activation,
            cuda=cfg.cuda
        )

        self.stims = None
        self.brain_data = None

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
            self.log_dir = os.path.join(log_dir, cfg.cfg_str, str(self.uuid))
            os.makedirs(self.log_dir)
        else:
            self.log_dir = None

        self.round_thresh = 5
        # A list of up to round_thresh length
        #  Each item is a tuple (actuals, targets, trial_end, stim list, obs list)
        #   The list is trial_len length.
        #     Each item is a stim or obs
        self.saved_data = []

        self.reset_soft()

    def reset_soft(self):
        self.stims = []
        self.brain_data = []
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
            new_stim = self.en_epoch.forward(
                brain_data, loss_history, self.epoch_type == EpochType.VAL_EN
            )
        elif self.epoch_type in (EpochType.CPN, EpochType.VAL_CPN):
            new_stim = self.cpn_epoch.forward(
                brain_data, loss_history, self.epoch_type == EpochType.VAL_CPN
            )
        else:
            # For now: no other thing
            raise ValueError(self.epoch_type)

        self.stims.append(new_stim.detach())
        self.brain_data.append(brain_data)

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

            if self.epoch_type == EpochType.EN:
                new_save = (actuals, targets, trial_end, self.stims, self.brain_data)
                self.saved_data.append(new_save)

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

    def train_en_closed_loop(self, loss_history, next_is_validation):
        # Unpack, aka form a single 'actuals', 'targets', 'trial_end',
        #  and pack a list of concatenated stims.

        bdata_len = len(self.saved_data[0][4][0])
        trial_len = len(self.saved_data[0][4])

        is_validation = next_is_validation

        done = False
        checkpoint_eidx = 0
        en_is_ready = False
        while not en_is_ready and checkpoint_eidx < 1000:
            for bidx in range(self.round_thresh):
                actuals, targets, trial_end, stims, brain_data = self.saved_data[bidx]

                for tidx in range(trial_len):
                    bd = brain_data[tidx]
                    stim = stims[tidx]

                    self.en_epoch.forward(
                        bd, loss_history, is_validation=is_validation, stim=stim
                    )

                self.en_epoch.feedback(
                    actuals, targets, trial_end, loss_history, is_validation
                )

                if is_validation:
                    loss_history.report_val_last_result(
                        actuals,
                        targets,
                        update_task_loss=False,
                    )
                else:
                    loss_history.report_by_result(
                        actuals,
                        targets,
                        labels=None,
                        update_task_loss=False,
                    )

                (
                    self.en,
                    self.opt_en,
                    next_is_validation,
                    en_is_ready,
                    user_data,
                ) = self.en_epoch.finish(loss_history, is_validation, reused_data=True)

                loss_history.report_user_data(user_data)
                loss_history.log(g_logger, msg=user_data.msg)

                is_validation = next_is_validation

                if en_is_ready:
                    break

                checkpoint_eidx += 1

        self.saved_data = []
        self.en_epoch.reset_period()
        return en_is_ready

    def finish(self, loss_history):
        if self.epoch_type in (EpochType.EN, EpochType.VAL_EN):
            (
                self.en,
                self.opt_en,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.en_epoch.finish(loss_history, self.epoch_type == EpochType.VAL_EN)

            if en_is_ready:
                self.epoch_type = EpochType.CPN
                self.cpn_epoch.set_en(self.en, self.opt_en)

                for param in self.cpn.parameters():
                    param.requires_grad = True
                for param in self.en.parameters():
                    param.requires_grad = False

                self.saved_data = []

            elif next_is_validation:
                self.epoch_type = EpochType.VAL_EN
            else:
                self.epoch_type = EpochType.EN

            if len(self.saved_data) == self.round_thresh and not en_is_ready:
                en_is_ready = self.train_en_closed_loop(loss_history,
                    next_is_validation)

                if en_is_ready:
                    self.epoch_type = EpochType.CPN
                    self.cpn_epoch.set_en(self.en, self.opt_en)

                    for param in self.cpn.parameters():
                        param.requires_grad = True
                    for param in self.en.parameters():
                        param.requires_grad = False
                else:
                    self.epoch_type = EpochType.EN

            we_are_done = False

        else:
            (
                we_are_done,
                next_is_validation,
                en_is_ready,
                user_data,
            ) = self.cpn_epoch.finish(loss_history, self.epoch_type == EpochType.VAL_EN)

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

    def report(self, loss_history, mike):
        if self.log_dir is not None:
            if (loss_history.eidx % 50) == 0:
                log_path = os.path.join(self.log_dir, "log")

                with open(log_path, "w") as f:
                    json.dump(loss_history.render(), f)
                    f.flush()

                cpn_path = os.path.join(self.log_dir, "cpn.model")
                en_path = os.path.join(self.log_dir, "en.model")
                mike_path = os.path.join(self.log_dir, "mike.model")

                torch.save(self.cpn.state_dict(), cpn_path)
                torch.save(self.en.state_dict(), en_path)
                torch.save(mike.state_dict(), mike_path)
