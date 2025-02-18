import pdb

import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor


class ClassifierUpdater:
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop("classifier")
        self.optimizer_c = kwargs.pop("optimizer_c")
        self.device = kwargs.pop("device")
        self.loss = F.cross_entropy

    def get_batch(self, batch, device=None, non_blocking=True):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    def __call__(self, engine, batch):
        report = {}
        self.classifier.train()
        x, y = self.get_batch(batch, device=self.device)
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, y)
        self.optimizer_c.zero_grad()
        loss.backward()
        self.optimizer_c.step()
        report.update({"y_pred": y_pred.detach()})
        report.update({"y": y.detach()})
        report.update({"loss": loss.detach().item()})
        return report
