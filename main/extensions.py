import json
import os
import time
import pdb


def register_metrics(engine, metrics):
    for key in metrics:
        metrics[key].attach(engine, key)


def check_accuracy(engine):
    score = engine.state.metrics["accuracy"]
    return score


def check_FID(engine):
    score = engine.state.metrics["FID"]
    return -score


def log_training_results(engine, log, pbar):
    metrics = engine.state.metrics
    log_dict = {}
    lr = engine._process_function.opt.param_groups[0]["lr"]
    log_dict["lr"] = lr
    log_dict["epoch"] = engine.state.epoch
    log_dict["iteration"] = engine.state.iteration
    for m in metrics.keys():
        if m in ["y_pred", "y"]:
            continue
        log_dict[m] = metrics[m]
        msg += f"{m}: {metrics[m]:4f} "
    pbar.log_message(msg)
    log["running"].append(log_dict)


def log_training_results_mps(engine, log, pbar):
    metrics = engine.state.metrics
    log_dict = {}
    lr = engine._process_function.optimizer_c.param_groups[0]["lr"]
    log_dict["lr"] = lr
    log_dict["epoch"] = engine.state.epoch
    log_dict["elapsed_time"] = engine.state.times["EPOCH_COMPLETED"]
    msg = f"Training Results - Epoch: {engine.state.epoch} time: {log_dict['elapsed_time']} lr: {lr:.6f} "
    for m in metrics.keys():
        if m in ["y_pred", "y"]:
            continue
        log_dict[m] = metrics[m]
        msg += f"{m}: {metrics[m]:4f} "
    pbar.log_message(msg)
    log["running"].append(log_dict)


def log_validation_results(engine, evaluator, val_loader, test_loader, log, pbar, dist):
    # Do validation
    evaluator.run(val_loader)
    log_dict = log["running"][-1]
    metrics = evaluator.state.metrics
    log_dict["val_accuracy"] = metrics["accuracy"]
    log_dict["val_loss"] = metrics["loss"]
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
            engine.state.epoch, metrics["accuracy"], metrics["loss"]
        )
    )
    # Do test iff the best validation accuracy was updated
    # if log["best_val_accuracy"] < log_dict["val_accuracy"] or log_dict["val_loss"] < log["best_val_loss"]:
    if log_dict["val_loss"] < log["best_val_loss"]:
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        log["test_accuracy"] = metrics["accuracy"]
    log["best_val_accuracy"] = max(log["best_val_accuracy"], log_dict["val_accuracy"])
    log["best_val_loss"] = min(log["best_val_loss"], log_dict["val_loss"])
    dump_log(log, dist)


def log_gan_training_results(engine, log, pbar):
    metrics = engine.state.metrics
    log_dict = {}
    log_dict["epoch"] = engine.state.epoch
    log_dict["iteration"] = engine.state.iteration
    lr_gen = engine._process_function.optG.param_groups[0]["lr"]
    lr_dis = engine._process_function.optD.param_groups[0]["lr"]
    log_dict["lr_gen"] = lr_gen
    log_dict["lr_dis"] = lr_dis
    msg = f"Training Results - Iteration: {engine.state.iteration} "
    for m in metrics.keys():
        log_dict[m] = metrics[m]
        msg += f"{m}: {metrics[m]} "
    pbar.log_message(msg)
    log["running"].append(log_dict)


def log_gan_validation_results(engine, evaluator, loader, log, pbar, dist):
    evaluator.run(loader, max_epochs=1, epoch_length=1)
    log_dict = log["running"][-1]
    metrics = evaluator.state.metrics
    log_dict["FID"] = metrics["FID"]
    # is_mean, is_std = metrics["IS"]
    # log_dict["IS"] = float(is_mean)
    # kid_mean, kid_std = metrics["KID"]
    # log_dict["KID"] = float(kid_mean)
    pbar.log_message(
        "Validation Results - Iteration: {}  FID: {:.4f}".format(
            engine.state.iteration, metrics["FID"]
        )
    )
    log["best_FID"] = min(log["best_FID"], metrics["FID"])
    # log["best_IS"] = max(log["best_IS"], float(is_mean))
    # log["best_KID"] = min(log["best_KID"], float(kid_mean))
    dump_log(log, dist)


def log_feature_results(engine, evaluator, loader, log, pbar, dist):
    evaluator.run(loader, max_epochs=1, epoch_length=1)
    log_dict = log["running"][-1]
    metrics = evaluator.state.metrics
    log_dict["Gap"] = metrics["Gap"]
    log_dict["m_entropy"] = metrics["m_entropy"]
    log_dict["c_entropy"] = metrics["c_entropy"]
    pbar.log_message(
        "Validation Results - Epoch: {}  Gap: {:.4f}, H(z): {:.4f}, H(z|c): {:.4f}".format(
            engine.state.epoch, metrics["Gap"], metrics["m_entropy"], metrics["c_entropy"]
        )
    )


def dump_log(log, dist):
    with open(os.path.join(dist, "log"), "w") as f:
        json.dump(log, f, indent=2, sort_keys=True, separators=(",", ": "))


def load_log(path):
    with open(os.path.join(path, "log"), "r") as f:
        log = json.load(f)
    return log


def resume_training(engine, resume_epoch):
    engine.state.iteration = resume_epoch * len(engine.state.dataloader)
    engine.state.epoch = resume_epoch
