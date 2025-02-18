import copy
import logging
import math
import os
import pdb
import shutil
from collections import OrderedDict

import ignite.distributed as ignite_dist
import torch
from ignite.engine import Events
from ignite.handlers.param_scheduler import LRScheduler, PiecewiseLinear
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split

import util.yaml_utils as yaml_utils

logger = logging.getLogger()


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def set_learning_rate_scheduler(trainer, optimizer, optimizer_config, max_iter):
    if "lr_milestone" not in optimizer_config:
        return None
    milestone = optimizer_config["lr_milestone"]
    if milestone == "linear":
        init_lr = optimizer_config["args"]["lr"]
        lr_scheduler = PiecewiseLinear(optimizer, "lr", milestones_values=[(0, init_lr), (max_iter, 0)], save_history=True)
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    elif milestone == "cosine":
        warmup = optimizer_config["warmup"] if "warmup" in optimizer_config else 0
        lr_scheduler = LRScheduler(CosineAnnealingLR(optimizer, max_iter, num_warmup_steps=warmup))
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    else:
        gamma = optimizer_config["lr_drop_rate"]
        lr_scheduler = LRScheduler(MultiStepLR(optimizer, milestones=milestone, gamma=gamma), save_history=True)
        trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    return lr_scheduler


def CosineAnnealingLR(optimizer, max_iteration, num_warmup_steps=0, num_cycles=7.0 / 16.0, last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, max_iteration - num_warmup_steps))
        no_progress = min(1, max(no_progress, 0))
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))  # this is correct

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def create_result_dir(result_dir, config_path):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, "{}/{}".format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)


def load_models(model_config):
    model = yaml_utils.load_model(model_config["func"], model_config["name"], model_config["args"])
    if "pretrained" in model_config:
        state_dict = torch.load(model_config["pretrained"], map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model


def desugar_keys(old_gen_dict):
    gen_dict = OrderedDict()
    for key, value in old_gen_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len('module.'):]
            print(f"Rename G's {key} ==> {key[len('module.'):]} for loading")
        gen_dict[new_key] = value
    return gen_dict


def load_pretrained_gans(gen, dis, model_path, strict=True, with_dy=False, finetune=False):
    path_suffix = model_path.split("/")[-1]
    if path_suffix == "imagenet256":
        old_gen_dict = torch.load(f"{model_path}/G.pth", map_location="cpu")
        old_dis_dict = torch.load(f"{model_path}/D.pth", map_location="cpu")
        gen_dict = OrderedDict()
        for key, value in old_gen_dict.items():
            new_key = key
            if key.startswith("module."):
                new_key = key[len('module.'):]
                print(f"Rename G's {key} ==> {key[len('module.'):]} for loading")
            gen_dict[new_key] = value
        dis_dict = OrderedDict()
        for key, value in old_dis_dict.items():
            new_key = key
            if key.startswith("module."):
                new_key = key[len('module.'):]
                print(f"Rename D's {key} ==> {key[len('module.'):]} for loading")
            dis_dict[new_key] = value
    else:
        state_dict = torch.load(model_path, map_location="cpu")
        gen_dict = state_dict["model_gen"]
        dis_dict = state_dict["model_dis"]
    if with_dy:
        for key in list(dis_dict.keys()):
            if "l_y" in key or "embedding" in key or "embed" in key:
                del dis_dict[key]
    elif finetune:
        for key in list(gen_dict.keys()):
            if "embed" in key or "shared" in key:
                del gen_dict[key]
        for key in list(dis_dict.keys()):
            if "l_y" in key or "embedding" in key or "embed" in key:
                del dis_dict[key]
    missing_g, unexpected_g = gen.load_state_dict(gen_dict, strict=strict)
    print(f"All keys of G: {gen_dict.keys()}")
    print(f"Missing key for G: {missing_g}")
    print(f"Unexpected key for G: {unexpected_g}")
    missing_d, unexpected_d = dis.load_state_dict(dis_dict, strict=strict)
    print(f"All keys of D: {gen_dict.keys()}")
    print(f"Missing key for D: {missing_d}")
    print(f"Unexpected key for D: {unexpected_d}")
    return gen, dis


def load_loss_function(loss_config, suffix=None):
    loss_config_ = copy.deepcopy(loss_config)
    loss_name = loss_config_["name"]
    if suffix is not None:
        loss_config_["name"] = "_".join([loss_name, suffix])
    loss = yaml_utils.load_method(loss_config_)
    return loss


def make_optimizer(model, opt_config):
    # Select from https://pytorch.org/docs/stable/optim.html
    # NOTE: The order of the arguments for optimizers follows their definitions.
    opt = yaml_utils.load_optimizer(model, opt_config["algorithm"], args=opt_config["args"])
    return opt


def make_gan_optimizer(model, config):
    opt = yaml_utils.load_optimizer(model, config["algorithm"], args=config["args"])
    return opt


def reduce_dataset(use_ratio, dataset):
    assert 0.0 < use_ratio and use_ratio <= 1.0
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    full_size = len(dataset)
    use_size = int(full_size * use_ratio)
    dataset, _ = random_split(dataset, [use_size, full_size - use_size])
    logger.info(f"## Reduced dataset size from {full_size} to {len(dataset)}")
    torch.manual_seed(seed)
    return dataset


def setup_train_dataloaders(config):
    # Dataset
    train = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        train = reduce_dataset(ratio, train)
    # DataLoader
    train_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    return train_loader


def setup_dataloaders(config, transform=None):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        all_train_dataset = reduce_dataset(ratio, all_train_dataset)
    train_size = int(len(all_train_dataset) * config["train_val_split_ratio"])
    val_size = len(all_train_dataset) - train_size
    train, val = random_split(all_train_dataset, [train_size, val_size])
    test = yaml_utils.load_dataset(config["dataset"], test=True)
    val.transform = test.transform

    # DataLoader
    train_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    test_loader = DataLoader(
        test, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True
    )
    torch.manual_seed(seed)
    return train_loader, val_loader, test_loader


def setup_eval_dataloaders(config):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)  # Ensure fixed seed to randomly split datasets
    all_train_dataset = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        train_dataset = reduce_dataset(ratio, all_train_dataset)
    else:
        train_dataset = all_train_dataset
    train_size = int(len(train_dataset) * config["train_val_split_ratio"])
    val_size = len(train_dataset) - train_size
    train, val = random_split(train_dataset, [train_size, val_size])
    test = yaml_utils.load_dataset(config["dataset"], test=True)
    all_train_dataset.transform = test.transform

    # DataLoader
    train_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    test_loader = DataLoader(
        test, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True
    )
    fid_eval_loader = DataLoader(all_train_dataset, len(all_train_dataset), shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    torch.manual_seed(seed)
    return train_loader, val_loader, test_loader, fid_eval_loader


def setup_multitask_dataloaders(config):
    # Dataset
    seed = torch.seed()
    torch.manual_seed(torch.initial_seed())
    target_dataset = yaml_utils.load_dataset(config["dataset"])
    if "use_ratio" in config["dataset"]:
        ratio = float(config["dataset"]["use_ratio"])
        target_dataset = reduce_dataset(ratio, target_dataset)
    train_size = int(len(target_dataset) * config["train_val_split_ratio"])
    val_size = len(target_dataset) - train_size
    train, val = random_split(target_dataset, [train_size, val_size])
    source_dataset = yaml_utils.load_dataset(config["source_dataset"])
    test = yaml_utils.load_dataset(config["dataset"], test=True)
    val.transform = test.transform

    # DataLoader
    target_loader = DataLoader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    source_loader = DataLoader(
        source_dataset, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True)
    test_loader = DataLoader(
        test, config["batchsize"], shuffle=False, num_workers=config["num_worker"], pin_memory=True
    )
    torch.manual_seed(seed)
    return source_loader, target_loader, val_loader, test_loader


def setup_unsupervised_dataloaders(config):
    # Dataset
    u_dataset = yaml_utils.load_dataset(config["pseudo"])
    batchsize = config["ubatchsize"] if "ubatchsize" in config else config["batchsize"]
    # DataLoader
    u_loader = DataLoader(
        u_dataset,
        batchsize,
        shuffle=True,
        num_workers=config["num_worker"],
        drop_last=True,
        pin_memory=True,
    )
    return u_loader


def setup_distributed_dataloaders(config):
    if ignite_dist.get_local_rank() > 0:
        ignite_dist.barrier()

    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)
    all_train_dataset = yaml_utils.load_dataset(config["dataset"])
    train_size = int(len(all_train_dataset) * config["train_val_split_ratio"])
    val_size = len(all_train_dataset) - train_size
    train, val = random_split(all_train_dataset, [train_size, val_size])
    test = yaml_utils.load_dataset(config["dataset"], test=True)
    val.transform = test.transform

    if ignite_dist.get_local_rank() == 0:
        ignite_dist.barrier()

    # DataLoader
    train_loader = ignite_dist.auto_dataloader(
        train, config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True
    )
    val_loader = ignite_dist.auto_dataloader(val, config["batchsize"], shuffle=False, num_workers=config["num_worker"])
    test_loader = ignite_dist.auto_dataloader(
        test, config["batchsize"], shuffle=False, num_workers=config["num_worker"]
    )
    torch.manual_seed(seed)
    return train_loader, val_loader, test_loader


def setup_distributed_dataloaders_for_gans(config):
    if ignite_dist.get_local_rank() > 0:
        ignite_dist.barrier()

    # Dataset
    seed = torch.seed()
    torch.manual_seed(42)
    ds_config = config["dataset"]
    all_train_dataset = yaml_utils.load_dataset(ds_config)
    train = reduce_dataset(ds_config["use_ratio"], all_train_dataset) if "use_ratio" in config else all_train_dataset
    val_size = min(len(all_train_dataset), config["evalsize"])
    val, _ = random_split(all_train_dataset, [val_size, len(all_train_dataset) - val_size])

    if ignite_dist.get_local_rank() == 0:
        ignite_dist.barrier()

    # DataLoader
    train_loader = ignite_dist.auto_dataloader(
        train, batch_size=config["batchsize"], shuffle=True, num_workers=config["num_worker"], drop_last=True
    )
    val_loader = ignite_dist.auto_dataloader(val, batch_size=config["batchsize"], shuffle=False, num_workers=config["num_worker"])
    torch.manual_seed(seed)
    return train_loader, val_loader


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
