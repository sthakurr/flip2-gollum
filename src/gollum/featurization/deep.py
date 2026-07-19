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

    def get_embeddings(self, x, batch_size=64):
        x = x.to(dtype=torch.float32)
        self.llm = self.llm.to(dtype=torch.float32)

        n_points = x.size(0)
        ids_split = int(x.shape[-1] / 2)

        # Per-position consensus token id over the whole featurized set, for
        # mutation-site pooling (positions differing from it are the mutations).
        consensus_ids = None
        if self.pooling_method == "mutation":
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
