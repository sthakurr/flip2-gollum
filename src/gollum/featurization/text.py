from dataclasses import dataclass
import os
from typing import Optional
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

import numpy as np

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from functools import partial
import torch.nn.functional as F


from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

# from openai import OpenAI

# client = OpenAI()

from transformers import AutoTokenizer
from gollum.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool
from gollum.featurization.mutation import _consensus





@lru_cache(maxsize=None)
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=model).data[0].embedding
    )


def ada_embeddings(texts, model="text-embedding-ada-002"):
    """
    Get ADA embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model: Model name to use for embedding (default is "text-embedding-ada-002")
    :type model: str
    :return: NumPy array of ADA embeddings
    """
    get_embedding_with_model = partial(get_embedding, model=model)

    with ProcessPoolExecutor() as executor:
        embeddings = list(
            tqdm(
                executor.map(get_embedding_with_model, texts),
                total=len(texts),
                desc="Getting Embeddings",
            )
        )
    return np.array(embeddings)


def ada_embeddings_3(texts, model="text-embedding-3-small"):
    return ada_embeddings(texts, model=model)


from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5EncoderModel, T5Config
from transformers import LlamaModel, LlamaConfig



@dataclass
class ModelConfig:
    name: str
    config_class: Optional[any] = None
    model_class: Optional[any] = None
    dropout_field: str = "dropout_rate"


MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "google-t5/t5-base": ModelConfig("google-t5/t5-base", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
    "Rostlab/prot_t5_xl_uniref50": ModelConfig(
        "Rostlab/prot_t5_xl_uniref50",
        T5Config,
        T5EncoderModel,
    ),
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": ModelConfig(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        LlamaConfig,
        LlamaModel,
        "attn_dropout",
    ),
}

def _is_esmc_model(model_name: str) -> bool:
    name = model_name.lower()
    return "esmc" in name or "evolutionaryscale/esmc" in name


def _normalize_esmc_model_name(model_name: str) -> str:
    normalized = model_name.lower()
    if "600m" in normalized:
        return "esmc_600m"
    if "300m" in normalized:
        return "esmc_300m"
    return "esmc_600m"

def get_model_and_tokenizer(model_name: str, device: str='cuda'):

    if _is_esmc_model(model_name):
        from esm.models.esmc import ESMC

        esmc_name = _normalize_esmc_model_name(model_name)
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = ESMC.from_pretrained(esmc_name, device=torch_device).to(torch_device)
        tokenizer = model.tokenizer
        return model, tokenizer

    if "prot_t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name)
        setattr(config, model_config.dropout_field, 0)
        torch_dtype = torch.bfloat16 if "prot_t5" in model_name.lower() else torch.float32
        model = model_config.model_class.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            model_name, device_map=device, trust_remote_code=True
        )

    return model, tokenizer


def get_tokens(
    texts,
    model_name="WhereIsAI/UAE-Large-V1",
    batch_size=32,
    max_length=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(model_name, "for get tokens")

    if _is_esmc_model(model_name):
        from esm.tokenization import get_esmc_model_tokenizers

        tokenizer = get_esmc_model_tokenizers()
        token_batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            ids = encoded.get("input_ids")
            if ids is None:
                ids = encoded["sequence_tokens"]
            masks = encoded.get("attention_mask")
            if masks is None:
                masks = (ids != tokenizer.pad_token_id).long()

            pad_len = 512 - ids.size(1)
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=tokenizer.pad_token_id)
                masks = torch.nn.functional.pad(masks, (0, pad_len), value=0)

            token_batches.append(torch.cat([ids, masks], dim=1))

        all_encoded_inputs = torch.cat(token_batches, dim=0)
        return all_encoded_inputs.cpu().numpy()

    if "prot_t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
        # ProtT5 requires space-separated amino acids
        texts = [" ".join(list(seq)) for seq in texts]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_batches = []
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids_padded = pad_sequence(
        [torch.tensor(ids) for ids in encoded_input.input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks_padded = pad_sequence(
        [torch.tensor(mask) for mask in encoded_input.attention_mask],
        batch_first=True,
        padding_value=0,
    )
    all_encoded_inputs = torch.cat(
        [input_ids_padded, attention_masks_padded], dim=1
    )
    return all_encoded_inputs.cpu().numpy()


def get_huggingface_embeddings(
    texts,
    model_name="tiiuae/falcon-7b",
    wt_ref=False,
    max_length=512,
    batch_size=16,
    pooling_method="cls",
    prefix=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    normalize_embeddings=False,
    use_cache=True,
):
    """
    General function to get embeddings from a HuggingFace transformer model. \
    Takes in a wt_ref argument to optionally use a wild-type sequence 
    as a reference to each input, which can help with small-mutation protein datasets. \

    Embeddings depend only on (sequences, model, pooling, prefix, max_length,
    normalize) — not on the BO seed — so they are cached to disk and reused
    across seeds/modes/re-runs. Inference uses bf16 autocast on CUDA for speed.
    """
    if max_length is None:
        max_length = 512
    texts = list(texts)
    wt_seq = _consensus(texts) if wt_ref else None
    cache = (
        _cache_path(
            f"hf_{model_name}_{pooling_method}_L{max_length}"
            f"_norm{int(normalize_embeddings)}_pre{prefix}_wtref{int(wt_ref)}",
            texts,
        )
        if use_cache
        else None
    )
    if cache and os.path.exists(cache):
        print(f"loading cached embeddings from {cache}")
        return np.load(cache)

    print(f"featurizing with {model_name}")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    left_padding = tokenizer.padding_side == "left"
    model.eval()

    # ProtT5 requires space-separated amino acids
    if "prot_t5" in model_name.lower():
        texts = [" ".join(list(seq)) for seq in texts]
        if wt_seq is not None:
            wt_seq = " ".join(list(wt_seq))

    # optionally add prefix to each text
    if prefix:
        texts = [prefix + text for text in texts]

    pooling_functions = {
        "average": average_pool,
        "cls": lambda x, _: x[:, 0],
        "last_token_pool": partial(last_token_pool, left_padding=left_padding),
        "weighted_average": weighted_average_pool,
    }

    autocast_enabled = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
    embeddings_list = []

    def _forward_pooled(batch_texts):
        """Tokenize, run the encoder, and pool a batch of sequences to ``(B, D)``."""
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            outputs = (
                model.encoder(**encoded_input)
                if is_encoder_decoder
                else model(**encoded_input)
            )
            return pooling_functions[pooling_method](
                outputs.last_hidden_state, encoded_input["attention_mask"]
            )

    wt_pooled = _forward_pooled([wt_seq]) if wt_ref else None

    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            pooled = _forward_pooled(batch_texts)
            if wt_ref:
                pooled = wt_pooled - pooled  # WT - mutant, (1, D) - (B, D)
            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
        # bf16 -> float32 (numpy has no bf16); also frees the autocast graph.
        embeddings_list.append(pooled.float().cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    if cache:
        np.save(cache, embeddings)
        print(f"cached embeddings to {cache}")
    return embeddings


def get_sentence_transformer_embeddings(
    texts, model_name="bigscience/sgpt-bloom-7b1-msmarco", batch_size=32
):
    model = SentenceTransformer(model_name)
    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        embeddings = model.encode(batch_texts)
        embeddings_list.append(embeddings)

    return np.concatenate(embeddings_list, axis=0)


def instructor_embeddings(
    texts,
    model_name="hkunlp/instructor-xl",
    instruction="Represent the chemistry procedure: ",
    normalize=False,
):
    """
    Get Instructor embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :param instruction: Instruction string for the embedding task
    :type instruction: str
    :return: NumPy array of Instructor embeddings
    """
    # Load the INSTRUCTOR model
    model = INSTRUCTOR(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 32
    sentence_embeddings_list = []
    paired_texts = [[instruction, text] for text in texts]

    for i in tqdm(range(0, len(paired_texts), batch_size)):
        batch_embeddings = model.encode(
            paired_texts[i : i + batch_size], normalize_embeddings=normalize
        )
        sentence_embeddings_list.append(batch_embeddings)


    return np.concatenate(sentence_embeddings_list, axis=0)


# ---------------------------------------------------------------------------
# ESM-C (EvolutionaryScale ESM Cambrian) embeddings + sparse-autoencoder feats
# ---------------------------------------------------------------------------

EMBEDDING_CACHE_DIR = os.environ.get("GOLLUM_EMBEDDING_CACHE", "embeddings_cache")


def _cache_path(tag, texts):
    """Deterministic cache file for a (tag, sequence-set) pair."""
    import hashlib

    h = hashlib.md5(("\n".join(texts)).encode()).hexdigest()[:16]
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    safe_tag = tag.replace("/", "_")
    return os.path.join(EMBEDDING_CACHE_DIR, f"{safe_tag}_{h}.npy")


def _esmc_pool(hidden, attn, pooling_method):
    """Pool ESM-C per-residue hidden states ``(B, L, D)`` to ``(B, D)``.

    ``attn`` is the ``(B, L)`` attention mask. For average pooling we mean over
    real residues, excluding padding *and* the BOS (pos 0) / EOS (last real
    token) special tokens — matching the single-sequence ``[1:-1]`` behavior.
    """
    if pooling_method == "cls":
        return hidden[:, 0, :]
    if pooling_method != "average":
        raise ValueError(f"Unsupported pooling_method: {pooling_method}")

    mask = attn.clone().bool()
    mask[:, 0] = False  # BOS
    lengths = attn.sum(dim=1)  # number of real tokens per sequence
    eos_idx = (lengths - 1).clamp(min=0)
    mask[torch.arange(mask.shape[0], device=mask.device), eos_idx] = False  # EOS
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def get_esmc_embeddings(
    texts,
    model_name="esmc_300m",
    pooling_method="average",
    batch_size=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    normalize_embeddings=False,
    use_cache=True,
):
    """Mean-pooled ESM-C (ESM Cambrian) per-sequence embeddings.

    Uses the EvolutionaryScale ``esm`` SDK. Batches the forward pass (tokenize →
    single batched model forward → masked mean-pool) under bf16 autocast for
    speed; falls back to the per-sequence SDK API if the batched/low-level API
    isn't available in the installed ``esm`` version. Results are cached to disk
    keyed by model + pooling + normalize + sequence set, since ESM-C is expensive.
    Raises a clear ImportError if ``esm`` is missing so the sweep can skip ESM-C.
    """
    texts = list(texts)
    cache = (
        _cache_path(f"esmc_{model_name}_{pooling_method}_norm{int(normalize_embeddings)}", texts)
        if use_cache
        else None
    )
    if cache and os.path.exists(cache):
        print(f"loading cached ESM-C embeddings from {cache}")
        return np.load(cache)

    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as e:
        raise ImportError(
            "ESM-C embeddings require the EvolutionaryScale `esm` package "
            "(`pip install esm`). Install it or drop the esmc_* arms from the "
            f"sweep. Original error: {e}"
        )

    print(f"featurizing with ESM-C ({model_name})")
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()
    autocast_enabled = device == "cuda" or (
        hasattr(device, "type") and device.type == "cuda"
    )

    def _normalize_and_np(pooled):
        if normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.float().cpu().numpy()

    def _batched():
        out_list = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
        ):
            batch = texts[i : i + batch_size]
            tok = client.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids = tok["input_ids"].to(device)
            attn = tok["attention_mask"].to(device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                model_out = client(sequence_tokens=input_ids)
                pooled = _esmc_pool(model_out.embeddings, attn, pooling_method)
            out_list.append(_normalize_and_np(pooled))
        return np.concatenate(out_list, axis=0)

    def _single():
        # Fallback: per-sequence SDK API (slow, no batching).
        out_list = []
        for seq in tqdm(texts, desc=f"Processing with {model_name} (single)"):
            protein = ESMProtein(sequence=seq)
            with torch.inference_mode():
                tensor = client.encode(protein)
                out = client.logits(
                    tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                emb = out.embeddings  # (1, L+special, d)
                if pooling_method == "average":
                    pooled = emb[0, 1:-1, :].mean(dim=0, keepdim=True)
                elif pooling_method == "cls":
                    pooled = emb[0, 0:1, :]
                else:
                    raise ValueError(f"Unsupported pooling_method: {pooling_method}")
            out_list.append(_normalize_and_np(pooled))
        return np.concatenate(out_list, axis=0)

    try:
        embeddings = _batched()
    except Exception as e:
        print(
            f"ESM-C batched forward failed ({type(e).__name__}: {e}); "
            "falling back to slower per-sequence featurization."
        )
        embeddings = _single()

    if cache:
        np.save(cache, embeddings)
    return embeddings


def get_esmc_sae_features(
    texts,
    model_name="esmc_300m",
    pooling_method="average",
    sae_weights_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Sparse-autoencoder features over ESM-C embeddings (InterPLM-style).

    Loads a trained SAE (a state dict containing encoder weights/bias) from
    ``sae_weights_path``, encodes the (mean-pooled) ESM-C embeddings into the
    sparse feature space, and prunes features that are never active or constant
    across the dataset (keeping only active & variable features, as planned).

    Raises a clear error if no weights path is provided / loadable so the SAE
    arm is skipped rather than crashing the sweep.
    """
    if sae_weights_path is None or not os.path.exists(str(sae_weights_path)):
        raise FileNotFoundError(
            "ESM-C SAE features require `sae_weights_path` pointing to trained "
            "SAE weights (e.g. InterPLM-style). None provided / file missing: "
            f"{sae_weights_path}. Provide weights or drop the esmc_sae arm."
        )

    embeddings = get_esmc_embeddings(
        texts, model_name=model_name, pooling_method=pooling_method, device=device
    )
    x = torch.from_numpy(embeddings).float().to(device)

    state = torch.load(sae_weights_path, map_location=device)
    # Support either a raw state_dict or a checkpoint wrapping one.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Find the encoder weight/bias by common naming conventions.
    w_key = next((k for k in state if k.endswith("encoder.weight") or k == "W_enc"), None)
    b_key = next((k for k in state if k.endswith("encoder.bias") or k == "b_enc"), None)
    if w_key is None:
        raise KeyError(
            f"Could not find an encoder weight in SAE checkpoint keys: {list(state)}"
        )
    W = state[w_key].to(device).float()
    # Linear weight is stored (out, in); W_enc-style is (in, hidden).
    pre = x @ W.T if W.shape[1] == x.shape[1] else x @ W
    if b_key is not None:
        pre = pre + state[b_key].to(device).float()
    feats = F.relu(pre).cpu().numpy()

    # Keep only features that are active (non-zero somewhere) and variable.
    keep = feats.std(axis=0) > 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"SAE: dropping {n_dropped} inactive/constant features, keeping {int(keep.sum())}")
    return feats[:, keep].astype(np.float32)
















