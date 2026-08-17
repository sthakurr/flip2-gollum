import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List
from peft import LoraConfig, get_peft_model
from gollum.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool
from gollum.featurization.text import get_model_and_tokenizer, esmc_pool
from gollum.featurization.utils.layers import get_target_layers
from torch.nn import init
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def token_mutation_pool(hidden, input_ids, consensus_ids, attn_mask):
    """Average hidden states only at token positions that differ from the consensus.

    ``consensus_ids`` is the per-position most-frequent token id over the featurized
    set; positions equal to it (special tokens, padding, and unmutated residues are
    all constant across combinatorial mutants) are excluded. Rows with no mutated
    positions fall back to a mask-averaged pool. Works for both HF ESM2
    (last_hidden_state) and ESM-C (embeddings) token layouts.
    """
    mask = (input_ids != consensus_ids.unsqueeze(0)) & attn_mask.bool()  # (B, T)
    m = mask.unsqueeze(-1).to(hidden.dtype)
    cnt = m.sum(dim=1)  # (B, 1)
    pooled = (hidden * m).sum(dim=1) / cnt.clamp_min(1.0)
    nomut = (cnt.squeeze(-1) == 0)
    if nomut.any():
        am = attn_mask[nomut].unsqueeze(-1).to(hidden.dtype)
        pooled[nomut] = (hidden[nomut] * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)
    return pooled


def token_mutation_context_topk_pool(
    hidden,
    input_ids,
    consensus_ids,
    attn_mask,
    top_k=4,
    locality=0.5,
    temperature=1.0,
):
    """Differentiable mutation-conditioned sparse context pooling.

    Each mutation token is used as a query over the other valid tokens. Context
    tokens are ranked by cosine similarity minus a logarithmic sequence-distance
    penalty; the mutation token and the ``top_k`` highest-ranked context tokens
    are then softmax-pooled. Per-mutation context vectors are averaged, yielding
    the same ``(batch, hidden_dim)`` output as :func:`token_mutation_pool`.

    ``top_k=0`` exactly reduces to mutation-only pooling. Hard top-k controls
    membership while gradients still flow through the selected hidden states and
    their softmax scores into a trainable encoder or its LoRA adapters.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}.")
    if locality < 0:
        raise ValueError(f"locality must be non-negative, got {locality}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    mutation_mask = (input_ids != consensus_ids.unsqueeze(0)) & attn_mask.bool()
    normalized = F.normalize(hidden.float(), p=2, dim=-1)
    out = []

    for batch_idx in range(hidden.shape[0]):
        mutation_indices = mutation_mask[batch_idx].nonzero(as_tuple=True)[0]
        valid_indices = attn_mask[batch_idx].bool().nonzero(as_tuple=True)[0]
        mutation_contexts = []

        for mutation_idx_tensor in mutation_indices:
            mutation_idx = int(mutation_idx_tensor.item())
            other_indices = valid_indices[valid_indices != mutation_idx]
            selected_indices = mutation_idx_tensor.view(1)
            selected_scores = hidden.new_tensor([1.0])

            if top_k and other_indices.numel():
                query = normalized[batch_idx, mutation_idx]
                similarities = normalized[batch_idx, other_indices] @ query
                distances = (other_indices - mutation_idx).abs().to(similarities.dtype)
                scores = similarities - locality * torch.log1p(distances)
                k = min(top_k, scores.numel())
                top = scores.topk(k)
                selected_indices = torch.cat([selected_indices, other_indices[top.indices]])
                selected_scores = torch.cat(
                    [selected_scores, top.values.to(selected_scores.dtype)]
                )

            weights = torch.softmax(selected_scores / temperature, dim=0).to(hidden.dtype)
            context = (
                hidden[batch_idx, selected_indices] * weights.unsqueeze(-1)
            ).sum(dim=0)
            mutation_contexts.append(context)

        if mutation_contexts:
            out.append(torch.stack(mutation_contexts).mean(dim=0))
        else:
            mask = attn_mask[batch_idx].unsqueeze(-1).to(hidden.dtype)
            out.append(
                (hidden[batch_idx] * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
            )

    return torch.stack(out)

class BaseNNFeaturizer(nn.Module):
    """
    Base class for neural network-based featurizers.
    Combines nn.Module functionality with the BaseFeaturizer interface.
    
    This is specifically for featurizers that need neural network capabilities, such as LLM-based featurizers.
    """
    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
    
    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the featurizer.
        
        Returns:
            int: Output dimension
        """
        return self._output_dim
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        pass


class ProjectionLayer(BaseNNFeaturizer):
    def __init__(
        self,
        input_dim: int = 3584,
        projection_dim: int = 64,
    ):
        super().__init__(input_dim=input_dim, projection_dim=projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim, projection_dim)

        self.fc1.bias.data.fill_(0.01)
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        return x


class LLMFeaturizer(BaseNNFeaturizer):
    def __init__(
        self,
        model_name: str = "WhereIsAI/UAE-Large-V1",
        input_dim: int = 1024,
        projection_dim: Optional[int] = None,
        trainable: bool = True,
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        lora_dropout: float = 0.2,
        lora_r: int = 4,
        lora_alpha: int = 16,
        modules_to_save: Optional[List[str]] = ["head"],
        target_ratio: float = 0.25,
        from_top: bool = True,
        gradient_checkpointing: bool = True,
        mutation_top_k: int = 4,
        mutation_locality: float = 0.5,
        mutation_temperature: float = 1.0,
    ):
        super().__init__(input_dim=input_dim, projection_dim=projection_dim)
        print(model_name, "for LLM")
        # ESM-C (EvolutionaryScale) is not a HuggingFace PreTrainedModel: it has a
        # different forward signature (sequence_tokens) and output field
        # (.embeddings), and needs manual gradient checkpointing.
        self.uses_esmc = "esmc" in model_name.lower()
        self.gradient_checkpointing = gradient_checkpointing
        self.llm, self.tokenizer = get_model_and_tokenizer(model_name, "cuda")
        if trainable:
            target_modules = get_target_layers(
                self.llm, target_ratio, from_top
            )

            self.llm = get_peft_model(
                self.llm,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_rslora=True,
                    modules_to_save=modules_to_save,
                ),
            )
            # Gradient checkpointing: recompute LLM activations during backward
            # instead of storing them. The joint DeepGP fit backprops through the
            # LLM for ALL train points at once (the GP needs the full covariance),
            # so activation memory scales with the train-set size and OOMs on a
            # large LLM. enable_input_require_grads is required for checkpointing
            # to propagate gradients through a LoRA-wrapped frozen backbone.
            if gradient_checkpointing and not self.uses_esmc:
                # HF-native gradient checkpointing (ESM-C uses manual
                # torch.utils.checkpoint in get_embeddings instead).
                try:
                    self.llm.enable_input_require_grads()
                    try:
                        self.llm.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    except TypeError:  # older transformers without the kwargs arg
                        self.llm.gradient_checkpointing_enable()
                except AttributeError:
                    print(
                        f"Gradient checkpointing not supported by {model_name}; "
                        "skipping (fit may use more memory)."
                    )
            self.llm.print_trainable_parameters()
        else:
            self.llm.requires_grad_(False)

        self.trainable = trainable
        self.embedding_dim = input_dim
        self.pooling_method = pooling_method
        self.mutation_top_k = mutation_top_k
        self.mutation_locality = mutation_locality
        self.mutation_temperature = mutation_temperature
        # Learned per-position pooling weights, materialised lazily on the first
        # forward (sequence length is only known once we see the tokenised input).
        self.pool_logits = None
        self.normalize_embeddings = normalize_embeddings
        self.input_dim = input_dim

        if projection_dim is not None:
            self.projector = ProjectionLayer(
                input_dim=input_dim, projection_dim=projection_dim
            )

        else:
            self.projector = nn.Identity()

        self.llm = self.llm.to(
            device=torch.device("cuda"), dtype=torch.float32
        )
        self.projector = self.projector.to(
            device=torch.device("cuda"), dtype=torch.float32
        )

    def get_embeddings(self, x, batch_size=16):
        x = x.to(dtype=torch.float32)
        self.llm = self.llm.to(dtype=torch.float32)

        n_points = x.size(0)
        ids_split = int(x.shape[-1] / 2)

        # Per-position consensus token id over the whole featurized set, for
        # mutation-site pooling (positions differing from it are the mutations).
        consensus_ids = None
        if self.pooling_method in ("mutation", "mutation_context_topk"):
            consensus_ids = x[:, :ids_split].long().mode(dim=0).values

        embedding_chunks = []

        current_idx = 0
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            input_ids = x[start_idx:end_idx, :ids_split].long()
            attn_mask = x[start_idx:end_idx, ids_split:].long()

            _config = getattr(self.llm, "config", None)
            is_enc_dec = getattr(_config, "is_encoder_decoder", False)

            def run(ids, mask):
                if self.uses_esmc:
                    return self.llm(sequence_tokens=ids, sequence_id=None)
                if is_enc_dec:
                    return self.llm.encoder(input_ids=ids, attention_mask=mask)
                return self.llm(input_ids=ids, attention_mask=mask)

            if self.trainable:
                if self.uses_esmc and self.gradient_checkpointing:
                    outputs = grad_checkpoint(
                        lambda t: run(t, attn_mask), input_ids, use_reentrant=False
                    )
                else:
                    outputs = run(input_ids, attn_mask)
            else:
                self.llm.eval()
                with torch.no_grad():
                    outputs = run(input_ids, attn_mask)

            last_hidden_state = (
                outputs.embeddings if self.uses_esmc else outputs.last_hidden_state
            )

            if self.pooling_method == "mutation":
                pooled = token_mutation_pool(
                    last_hidden_state, input_ids, consensus_ids, attn_mask
                )
            elif self.pooling_method == "mutation_context_topk":
                pooled = token_mutation_context_topk_pool(
                    last_hidden_state,
                    input_ids,
                    consensus_ids,
                    attn_mask,
                    top_k=self.mutation_top_k,
                    locality=self.mutation_locality,
                    temperature=self.mutation_temperature,
                )
            elif self.pooling_method == "average":
                pooled = (esmc_pool(last_hidden_state, attn_mask, "average")
                          if self.uses_esmc
                          else average_pool(last_hidden_state, attn_mask))
            elif self.pooling_method == "cls":
                pooled = last_hidden_state[:, 0]
            elif self.pooling_method == "last_token_pool":
                pooled = last_token_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "weighted_average":
                pooled = weighted_average_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "learned":
                if self.pool_logits is None:
                    self.pool_logits = nn.Parameter(
                        torch.zeros(
                            last_hidden_state.size(1),
                            device=last_hidden_state.device,
                            dtype=last_hidden_state.dtype,
                        )
                    )
                logits = self.pool_logits.unsqueeze(0).masked_fill(
                    ~attn_mask.bool(), float("-inf")
                )
                alpha = torch.softmax(logits, dim=1)
                pooled = torch.einsum("bt,btd->bd", alpha, last_hidden_state)
            else:
                raise ValueError(
                    f"Unknown pooling method: {self.pooling_method}"
                )

            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

            batch_size = pooled.size(0)
            embedding_chunks.append(pooled.to(dtype=torch.float64))
            current_idx += batch_size
            del outputs, last_hidden_state, pooled
        
        embeddings = torch.cat(embedding_chunks, dim=0)
        return embeddings

    def forward(self, x):

        # case because of botorch acquisition function
        if x.dim() == 3:

            n_candidates, n_train, d = x.shape
            train_data = x[0, : n_train - 1, :]
            # TODO update when batch
            all_candidates = x[:, n_train - 1, :]
            with torch.no_grad():

                train_embeddings = self.get_embeddings(train_data)
                all_candidate_embeddings = self.get_embeddings(all_candidates)

            train_embeddings = train_embeddings.unsqueeze(0).expand(
                n_candidates, -1, -1
            )
            candidate_embeddings = all_candidate_embeddings.unsqueeze(1)
            embeddings = torch.cat(
                [train_embeddings, candidate_embeddings], dim=1
            )

        elif x.dim() == 2:
            embeddings = self.get_embeddings(x)

        return self.projector(embeddings)

    @property
    def output_dim(self):
        return (
            self.projector[-1].out_features
            if isinstance(self.projector, nn.Sequential)
            else self.embedding_dim
        )
