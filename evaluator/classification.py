import torch
from ignite.utils import convert_tensor


class ClassifierEvaluator:
    def __init__(self, *args, **kwargs):
        self.classifier = kwargs.pop("classifier")
        self.device = kwargs.pop("device")

    def get_batch(self, batch, device=None):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=True),
            convert_tensor(y, device=device, non_blocking=True),
        )

    def __call__(self, engine, batch):
        classifier = self.classifier
        classifier.eval()
        x, y = self.get_batch(batch, device=self.device)
        with torch.no_grad():
            y_pred = classifier(x, test=True)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        return (y_pred.detach(), y.detach())
