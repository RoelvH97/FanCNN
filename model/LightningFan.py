# import necessary modules
import lightning.pytorch as pl
import numpy as np
import sys

from torch.optim import *
from torch.optim.lr_scheduler import *
from utils import region_growing
from .FanCNN import *


class LightningFan(pl.LightningModule):
    """
    Lightning-style trainer for FanCNN
    """
    def __init__(self, config):
        super().__init__()
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]

        self.loss = self.str_to_attr(self.c_m["loss"])(**self.c_m["loss_kwargs"])
        self.model = self.str_to_attr(self.c_m["name"])(self.c_m)

        # for correlation measures
        self.v_pred = []
        self.v_true = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.str_to_attr(self.c_o["name"])(self.model.parameters(), **self.c_o["OPTIMIZER"])
        scheduler = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["SCHEDULER"])
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)

        loss_dict = self.loss(y_pred, y_true)
        self.log_loss(loss_dict, "train")
        return sum(loss_dict.values())

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)

        loss_dict = self.loss(y_pred, y_true)
        self.log_loss(loss_dict, "val")

        self.v_pred.append(self.polar_to_volume(y_pred))
        self.v_true.append(self.polar_to_volume([y_true[0], y_true[2], y_true[3]]))
        return sum(loss_dict.values())

    def on_validation_epoch_end(self) -> None:
        volume_corr = self.correlate(self.v_pred, self.v_true)
        for key, value in volume_corr.items():
            self.log(f"val/corr_{key}", value, on_epoch=True, prog_bar=True, logger=True)

        # reset
        self.v_pred = []
        self.v_true = []

    def log_loss(self, losses, mode):
        for key, value in losses.items():
            self.log(f"{mode}/loss_{key}", value, on_epoch=True, prog_bar=True, logger=True)

    def polar_to_volume(self, input):
        r_l, r_cp, r_ncp = input[:3]

        r_l = torch.squeeze(r_l)
        r_cp = torch.squeeze(r_cp)
        r_ncp = torch.squeeze(r_ncp)
        r_l[r_l < 0] = 0
        r_cp[r_cp < 0] = 0
        r_ncp[r_ncp < 0] = 0

        if len(input) == 3:
            r_cp, r_ncp = r_l + r_cp + r_ncp, r_l + r_ncp
        elif len(input) == 4:
            # classifier attention
            cls = input[3]
            cls = np.argmax(np.squeeze(F.softmax(cls, dim=1).detach().cpu().data.numpy()), axis=0)

            ncp = region_growing((cls == 2) | (cls == 3), r_ncp.detach().cpu().data.numpy() >= 1.5)
            ncp = torch.tensor(ncp, device=r_l.device)

            r_cp = r_l + r_ncp + r_cp
            r_ncp = r_l + r_ncp * ncp

        volumes = []
        for r in (r_l, r_cp, r_ncp):
            r *= self.c_m["dr"]  # to mm
            roll = torch.roll(r, 1, 0)

            # calculate volume
            v = 0.5 * self.c_m["dz"] * r * roll * (
                torch.tensor(2 / r.shape[1] * np.pi, device=r.device))  # 0.5 (SAS triangle) * 0.5 mm
            volumes.append(v.sum().item())  # mm^3

        volumes[1] -= volumes[2]
        volumes[2] -= volumes[0]
        return volumes

    @staticmethod
    def correlate(inputs, targets):
        icc_l = np.corrcoef(np.array(inputs)[:, 0], np.array(targets)[:, 0])[0, 1]
        icc_cp = np.corrcoef(np.array(inputs)[:, 1], np.array(targets)[:, 1])[0, 1]
        icc_ncp = np.corrcoef(np.array(inputs)[:, 2], np.array(targets)[:, 2])[0, 1]
        if icc_l != icc_l:
            icc_l = 0.
        if icc_cp != icc_cp:
            icc_cp = 0.
        if icc_ncp != icc_ncp:
            icc_ncp = 0.

        return {"l": icc_l, "cp": icc_cp, "ncp": icc_ncp}

    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)
