"""Config assembly for BO runs: CLI overrides, validation, acquisition specs
and W&B run naming.
"""
CHECKPOINT_DIR = "/scratch/saumya/gollum/"

MODEL_EMBEDDING_SIZES = {
    "WhereIsAI/UAE-Large-V1": 1024,
    "nomic-ai/modernbert-embed-base": 768,
    "Qwen/Qwen2-7B-Instruct": 3584,
    "t5-base": 768,
    "mistralai/Mistral-7B-Instruct-v0.2": 4096,
    "text-embedding-3-large": 3072,
    "nomic-ai/modernbert-embed-base;get_huggingface_embeddings;normalize:False;pooling:cls": 768,
    "Qwen/Qwen2-7B-Instruct;get_huggingface_embeddings;normalize:False;pooling:last_token": 3584,
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": 768,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "Rostlab/prot_t5_xl_uniref50": 1024,
}


def configure_embedding_size(config, model_name):
    if model_name in MODEL_EMBEDDING_SIZES:
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "input_dim"
        ] = MODEL_EMBEDDING_SIZES[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in supported models.")
    return config


def configure_pooling_method(config, model_name):
    hugging_face_models = {
        "nomic-ai/modernbert-embed-base": "cls",
        "mixedbread-ai/mxbai-embed-large-v1": "cls",
        "WhereIsAI/UAE-Large-V1": "cls",
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct": "last_token_pool",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm": "average",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm-from-rxn": "average",
        "t5-base": "average",
        "Qwen/Qwen2-7B-Instruct": "last_token_pool",
        "facebook/esm2_t33_650M_UR50D": "average",
        "Rostlab/prot_t5_xl_uniref50": "average",
    }

    if model_name in hugging_face_models:
        config["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = (
            hugging_face_models[model_name]
        )
    else:
        raise ValueError(
            f"Model {model_name} not found in supported models. Please specify pooling method manually."
        )

    if (
        config["surrogate_model"]["class_path"]
        in ["gollum.surrogate_models.gp.DeepGP"]
    ):
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "pooling_method"
        ] = hugging_face_models[model_name]

    return config


def configure_benchmark_datasets(config):
    """Configure dataset settings based on benchmark name."""
    benchmark = config["benchmark"]
    if benchmark.startswith("bh"):
        reaction_num = benchmark[-1]

        config["data"]["init_args"][
            "data_path"
        ] = f"data/reactions/buchwald-hartwig/bh_reaction_{reaction_num}_procedure_template_basic.csv"

        config["data"]["init_args"]["target_column"] = "objective"
        config["data"]["init_args"]["maximize"] = True

    return config


def validate_configuration(config):
    """Validate that the configuration is consistent."""
    surrogate_class = config["surrogate_model"]["class_path"]

    featurizer_config = config["data"]["init_args"]["featurizer"]["init_args"]
    representation = featurizer_config.get("representation")

    # check for invalid configurations
    if surrogate_class == "gollum.surrogate_models.gp.GP" and representation == "get_tokens":
        raise ValueError("Standard GP or PLLM shouldn't use 'get_tokens'. This is for trainable LLM models only.")
    if surrogate_class == "gollum.surrogate_models.gp.DeepGP":
        ft_class = (
            config["surrogate_model"].get("init_args", {})
            .get("finetuning_model", {})
            .get("class_path", "")
        )
        is_llm_featurizer = "LLMFeaturizer" in ft_class
        if is_llm_featurizer and representation != "get_tokens":
            raise ValueError("DeepGP with LLMFeaturizer requires 'get_tokens' representation.")
        if not is_llm_featurizer and representation == "get_tokens":
            raise ValueError("DeepGP with ProjectionLayer requires pre-computed embeddings, not 'get_tokens'.")

    # The stuyver kernel's dimension-scaled priors assume inputs in [0, 1]^d
    if config["surrogate_model"].get("init_args", {}).get("kernel") == "stuyver":
        if surrogate_class != "gollum.surrogate_models.gp.GP":
            raise ValueError("kernel='stuyver' is only implemented for the static GP.")
        if config["data"]["init_args"]["normalize_input"] != "per_dim_normalisation":
            raise ValueError(
                "kernel='stuyver' requires normalize_input='per_dim_normalisation' "
                "(its priors are calibrated for per-dimension [0, 1] features)."
            )
        if config["data"]["init_args"].get("reduce_dim"):
            raise ValueError(
                "kernel='stuyver' with reduce_dim: PCA runs after normalization, so "
                "the features reaching the kernel are not in [0, 1]."
            )

    # Ensure model embedding sizes are correct
    model_name = featurizer_config.get("model_name")
    if model_name in MODEL_EMBEDDING_SIZES:
        embedding_size = MODEL_EMBEDDING_SIZES[model_name]

        if ("surrogate_model" in config and "init_args" in config["surrogate_model"]
                and "finetuning_model" in config["surrogate_model"]["init_args"]):
            ft_init_args = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]
            current_dim = ft_init_args.get("input_dim")
            if current_dim != embedding_size:
                print(f"Updating input_dim from {current_dim} to {embedding_size} for {model_name}")
                ft_init_args["input_dim"] = embedding_size

    return config


def acq_spec(acq):
    """Build an acquisition config from a one-word tag (--acq). beta is injected
    separately for ucb."""
    specs = {
        "ei": "botorch.acquisition.analytic.ExpectedImprovement",
        "logei": "botorch.acquisition.analytic.LogExpectedImprovement",
        "ucb": "botorch.acquisition.analytic.UpperConfidenceBound",
        "greedy": "botorch.acquisition.analytic.PosteriorMean",
    }
    if acq not in specs:
        raise ValueError(f"Unknown --acq '{acq}', expected one of {list(specs)}")
    return {"class_path": specs[acq], "init_args": {"maximize": True}}


def apply_cli_overrides(config):
    """Fold the flat CLI arguments into the nested config they override."""
    if config.get("benchmark", None) is not None:
        config = configure_benchmark_datasets(config)

    if config.get("data_path", None) is not None:
        config["data"]["init_args"]["data_path"] = config["data_path"]

    if config.get("kernel", None) is not None:
        config["surrogate_model"]["init_args"]["kernel"] = config["kernel"]

    if config.get("lora_r", None) is not None:
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["lora_r"] = config["lora_r"]

    if config.get("lora_dropout", None) is not None:
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["lora_dropout"] = config["lora_dropout"]

    if config.get("test_path", None) is not None:
        config["data"]["init_args"]["test_path"] = config["test_path"]

    if config.get("acq", None) is not None:
        config["acquisition"] = acq_spec(config["acq"])

    if config.get("beta", None) is not None and \
            "UpperConfidenceBound" in config["acquisition"]["class_path"]:
        config["acquisition"]["init_args"]["beta"] = config["beta"]

    return config


def make_run_name(config, mode):
    """Build a meaningful W&B run name. Uses an explicit ``name`` from the config
    if present (the representation arms set this), otherwise derives one from the
    featurizer representation / model / PCA so different arms don't collide."""
    base = config.get("name")
    if not base:
        feat = config["data"]["init_args"]["featurizer"]["init_args"]
        rep = feat.get("representation", "feat")
        model_name = feat.get("model_name")
        if rep in ("get_huggingface_embeddings", "get_esmc_embeddings"):
            base = (model_name or rep).split("/")[-1]
        elif rep == "get_esmc_sae_features":
            base = f"{(model_name or 'esmc').split('/')[-1]}_sae"
        elif rep == "onehot":
            base = "onehot"
        else:
            base = rep
        reduce_dim = config["data"]["init_args"].get("reduce_dim")
        if reduce_dim:
            base += f"_pca{reduce_dim}"
    if config["surrogate_model"].get("init_args", {}).get("kernel") == "stuyver":
        base += "_stuyver"
    acq = config.get("acq")
    beta = config.get("beta")
    if acq == "ucb" or (acq is None and beta is not None):
        acq_tag = f"_ucb{beta}"
    elif acq:
        acq_tag = f"_{acq}"
    else:
        acq_tag = ""
    return f"{base}{acq_tag}_{mode}_seed{config['seed']}"
