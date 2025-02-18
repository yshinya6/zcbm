import argparse
import functools
import os
import pdb
import random
import sys
import traceback

import ignite
import ignite.utils as ignite_utils
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from ignite.metrics.running_average import RunningAverage
from torch.backends import cudnn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import extensions

from evaluator.classification import ClassifierEvaluator
from util import train_util, yaml_utils


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("--------------")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("--------------")


def main(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    logger = ignite_utils.setup_logger(name=config["pattern"])
    log_basic_info(logger, config)

    # Create output directory
    logger.info("Creating output directory")
    out = os.path.join(config["results_dir"], config["pattern"], f"experiment{config['experiment_id']}")
    train_util.create_result_dir(out, config["config_path"])

    # Model
    logger.info("Constructing models")
    classifier = train_util.load_models(config["models"]["classifier"])
    classifier = torch.nn.DataParallel(classifier)
    classifier.to(device)

    # DataLoader
    logger.info("Constructing data loaders")
    config["dataset"]["args"]["transform"] = classifier.module.img_preprocess
    train_loader, val_loader, test_loader = train_util.setup_dataloaders(config)

    # Optimizer
    logger.info("Constructing optimizers")
    opt_c = train_util.make_optimizer(classifier, config["optimizer_c"])

    if config["resume"]:
        logger.info("Resume training with snapshot:{}".format(config["resume"]))
        if os.path.isfile(config["resume"]):
            checkpoint = torch.load(config["resume"])
            classifier.load_state_dict(checkpoint["model"])
            opt_c.load_state_dict(checkpoint["optimizer_c"])
            config["start_epoch"] = opt_c.param_groups[0]["epoch"]

    # Updater
    logger.info("Constructing updater and evaluators")
    max_iter = int(len(train_loader) * config["epoch"])
    kwargs = config["updater"]["args"] if "args" in config["updater"] else {}
    kwargs.update(
        {
            "classifier": classifier,
            "optimizer_c": opt_c,
            "device": device,
            "train_loader": train_loader,
        }
    )
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)

    # Trainer := Ignite.Engine
    trainer = Engine(updater)
    monitoring_metrics = ["train_loss"]

    # RunningAverage(src=Accuracy(output_transform=lambda x: (x["y_pred"], x["y"]))).attach(trainer, "train_accuracy")
    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "train_loss")
    logger.info(f"Monitoring Metrics: {monitoring_metrics}")
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # Evaluator for classifier
    evaluator = Engine(ClassifierEvaluator(classifier=classifier, device=device))
    eval_metrics = {"accuracy": Accuracy(), "loss": Loss(F.cross_entropy)}
    extensions.register_metrics(evaluator, eval_metrics)

    # Event Handlers
    logger.info("Constructing event handlers")
    # Log Handler
    log = {"running": [], "best_val_accuracy": 0.0, "best_val_loss": 10000000000000.0, "test_accuracy": 0.0}
    log = log if not config["resume"] else extensions.load_log(out)
    logger_train = functools.partial(extensions.log_training_results_mps, log=log, pbar=pbar)
    logger_val = functools.partial(
        extensions.log_validation_results,
        evaluator=evaluator,
        val_loader=val_loader,
        test_loader=test_loader,
        log=log,
        pbar=pbar,
        dist=str(out),
    )

    # Check Point Handler
    check_pointer = ModelCheckpoint(str(out), filename_prefix="model", n_saved=1)
    best_check_pointer = ModelCheckpoint(
        str(out), filename_prefix="best", score_function=extensions.check_accuracy, n_saved=1, score_name="val_accuracy"
    )

    # Learning Rate Schedule Handler
    lr_scheduler_c = train_util.set_learning_rate_scheduler(trainer, opt_c, config["optimizer_c"], max_iter)

    # Append handlers to trainer/evaluator engine
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger_val)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["snapshot_interval"]),
        check_pointer,
        {
            "model": classifier,
            "optimizer_c": opt_c,
            "lr_scheduler_c": lr_scheduler_c,
        },
    )
    evaluator.add_event_handler(
        Events.COMPLETED(every=config["snapshot_interval"]),
        best_check_pointer,
        {
            "model": classifier,
        },
    )

    if config["resume"]:
        resumer = functools.partial(extensions.resume_training, resume_epoch=config["start_epoch"])
        trainer.add_event_handler(Events.EPOCH_STARTED, resumer)

    # Run the training
    logger.info("Running train script")
    try:
        trainer.run(train_loader, max_epochs=config["epoch"])
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        log.update({"best_model": str(best_check_pointer.last_checkpoint)})
        extensions.dump_log(log, str(out))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/base.yml", help="path to config file")
    parser.add_argument("--results_dir", type=str, default="./result", help="directory to save the results to")
    parser.add_argument("--resume", type=str, default="", help="path to the snapshot")
    parser.add_argument("--experiment_id", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))
    config.config_dict["pattern"] = yaml_utils.make_pattern(config)
    config.config_dict.update(vars(args))
    main(config.config_dict)


if __name__ == "__main__":
    run()
