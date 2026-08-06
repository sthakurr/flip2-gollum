"""
Training script for Gollum BO.

Without --test_path: runs Phase-1 BO over the whole data_path pool.
With --test_path: runs Phase-1 BO on data_path, then Phase-2 BO over the test
set seeded with the Phase-1 points.
--test_spearman true additionally scores how well the Phase-1 model ranks the
test set (train->test generalisation); --phase2_iters 0 stops after that.

Usage:
  python train.py --config configs/flip2_arms/static/static_esm2_dense.yaml \
      --data_path data/flip2/nucB/two_to_many_train.csv \
      --test_path data/flip2/nucB/two_to_many_test.csv --seed 1
"""
from gollum.utils.runtime import configure_runtime

# Warning filters must be registered before the emitting libraries are imported,
# so the remaining imports deliberately sit below this call.
configure_runtime()

from botorch.acquisition import AcquisitionFunction  # noqa: E402
from jsonargparse import ActionConfigFile, ArgumentParser  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402

from gollum.bo.optimizer import BotorchOptimizer  # noqa: E402
from gollum.bo.runner import train  # noqa: E402
from gollum.data.module import BaseDataModule  # noqa: E402
from gollum.surrogate_models.gp import SurrogateModel  # noqa: E402


def main():
    # Initialize the parser with a description
    parser = ArgumentParser(
        description="Training script",
        default_config_files=[],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")

    # parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument(
        "--data_path", type=str, help="Phase-1 train data csv path"
    )
    parser.add_argument(
        "--test_path", type=str, default=None,
        help="Optional held-aside test csv; if given, runs gate + Phase-2 BO on it",
    )
    parser.add_argument(
        "--phase1_iters", type=int, help="Phase-1 (train) BO iterations (full mode)"
    )
    parser.add_argument(
        "--phase2_iters", type=int, help="Phase-2 (test) BO iterations (full mode)"
    )
    parser.add_argument(
        "--test_spearman", type=bool, default=False,
        help="After Phase-1, score the surrogate's ranking of the test set (spearman etc.)",
    )
    parser.add_argument(
        "--seed_source",
        type=str,
        default="phase1_bo",
        help="Phase-2 seed baseline: phase1_bo | random_train | all_train | none",
    )
    parser.add_argument(
        "--phase2_seed_size",
        type=int,
        help="Seed size for random_train / none (default: matched to Phase-1 budget)",
    )
    parser.add_argument("--acq", type=str, help="Acquisition override: logei | ucb | greedy")
    parser.add_argument("--beta", type=float, help="UCB beta override (acquisition.init_args.beta)")
    parser.add_argument("--lora_r", type=int, help="LoRA rank override for the finetuning featurizer")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout override for the finetuning featurizer")
    # parser.add_argument("--group", type=str, help="Wandb group runs")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name base")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name (default gollum-flip2-final)")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the finetuned model after training")
    parser.add_argument("--save_epoch_models", type=bool, default=False, help="DeepGP: save the finetuning model at every fit epoch")
    parser.add_argument("--visualize_latent", type=bool, default=False, help="DeepGP: plot latent-space (UMAP) evolution across fit epochs")
    parser.add_argument("--plot_distances", type=bool, default=False, help="DeepGP: plot d_hh/d_ll/d_hl (80/20 fitness-quantile latent distances) across fit epochs")

    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(SurrogateModel, "surrogate_model", instantiate=False)

    parser.add_subclass_arguments(
        AcquisitionFunction,
        "acquisition",
        instantiate=False,
        skip=["model", "best_f"],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    # parse arguments
    args = parser.parse_args()
    seed_everything(args["seed"], workers=True)
    train(args.as_dict())


if __name__ == "__main__":
    main()
