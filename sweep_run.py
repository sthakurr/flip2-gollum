import argparse
import yaml
from train import train, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_iters", type=int, required=True)
    parser.add_argument("--n_train_iter", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    config["data"]["init_args"]["train_path"] = f"data/flip2/{args.dataset}_train.csv"
    config["data"]["init_args"]["test_path"] = f"data/flip2/{args.dataset}_test.csv"
    config["n_iters"] = args.n_iters
    config["n_train_iter"] = args.n_train_iter
    config["seed"] = args.seed
    config["group"] = "sweep_top5"
    config["wandb_project"] = "gollum_flip2"

    seed_everything(args.seed)
    train(config)


if __name__ == "__main__":
    main()
