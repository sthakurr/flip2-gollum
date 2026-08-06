"""Run orchestration: build the data module and optimizer from a config, then
drive the phases (Phase-1 BO, test eval, Phase-2 BO) inside a single W&B run.
"""
import logging
import os

import torch
import wandb

from gollum.bo.config import (
    CHECKPOINT_DIR,
    apply_cli_overrides,
    make_run_name,
    validate_configuration,
)
from gollum.bo.optimizer import BotorchOptimizer
from gollum.bo.phases import run_bo, run_phase2, run_test_eval
from gollum.metrics import calculate_data_stats, log_data_stats
from gollum.utils.config import flatten, instantiate_class
from gollum.utils.runtime import pl_logger


def setup_data(config):
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=config["seed"]
    )
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    dm = instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )

    return dm


def setup_bo_optimizer(config):
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    bo = BotorchOptimizer(
        surrogate_model_config=surrogate_model_config,
        acq_function_config=acquisition_config,
        batch_strategy=bo_config["batch_strategy"],
        batch_size=bo_config["batch_size"],
        finetune_start_iter=bo_config.get("finetune_start_iter", 0),
    )
    return bo


def train(config):
    config = apply_cli_overrides(config)
    config = validate_configuration(config)
    wandb_config = flatten(config)

    # Two-phase iff a test set is configured. mode label is for naming/logging only
    # (analysis scripts filter W&B runs on config.mode).
    has_test = config["data"]["init_args"].get("test_path") is not None
    # test_spearman: score the Phase-1 model's ranking of the test set. Runs
    # after Phase-1 and is independent of whether Phase-2 follows.
    test_spearman = bool(config.get("test_spearman"))
    if test_spearman and not has_test:
        raise ValueError("--test_spearman needs a test set; provide --test_path.")
    mode = "full" if has_test else "bo"
    wandb_config["mode"] = mode
    run_name = make_run_name(config, mode)

    with wandb.init(
        project=config.get("wandb_project") or "gollum-flip2-final",
        config=wandb_config, name=run_name
    ) as run:

        dm = setup_data(config)
        bo = setup_bo_optimizer(config)

        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        # Phase iteration counts: two-phase uses separate phase1/phase2 budgets;
        # Phase-1-only falls back to n_iters.
        phase1_iters = config.get("phase1_iters") or config.get("n_iters", 3)
        # phase2_iters=0 -> Phase-1 (+ test eval) only.
        phase2_iters = config.get("phase2_iters")
        if phase2_iters is None:
            phase2_iters = config.get("n_iters", 3)

        if not has_test:
            # Phase-1 BO only, over the whole data_path pool.
            run_bo(config, dm, bo, data_stats, n_iters=phase1_iters)
        else:
            # `or` (not get-with-default): the CLI registers seed_source as
            # present-but-None when unset, so a plain .get would return None.
            seed_source = config.get("seed_source") or "phase1_bo"
            # phase1_bo seeds Phase-2 from the Phase-1-collected points, so it
            # runs Phase-1 first; the other baselines seed independently.
            if seed_source == "phase1_bo":
                run_bo(config, dm, bo, data_stats, n_iters=phase1_iters)
                if test_spearman:
                    run_test_eval(config, dm, bo)
            if phase2_iters > 0:
                run_phase2(
                    config, dm, bo, phase2_iters,
                    epoch_offset=phase1_iters + 1, seed_source=seed_source,
                )

        # Save finetuned model if using DeepGP
        if config["surrogate_model"]["class_path"] == "gollum.surrogate_models.gp.DeepGP" and config["save_model"] == True:
            model_save_path = os.path.join(
                CHECKPOINT_DIR, run.name if run else "default", "finetuned_model.pt"
            )
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(
                bo.surrogate_model.finetuning_model.state_dict(), model_save_path
            )
            print(f"Saved finetuned model to {model_save_path}")

        pl_logger.setLevel(logging.INFO)
        wandb.finish()
