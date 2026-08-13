import torch

from gollum.featurization.deep import (
    token_mutation_context_topk_pool,
    token_mutation_pool,
)
from gollum.featurization.text import (
    _mutation_context_topk_pool,
    _mutation_pool,
)


def test_static_topk_zero_matches_mutation_pool():
    hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    attention = torch.ones(2, 6, dtype=torch.long)
    sequences = ["ACWE", "ACDE"]
    consensus = "ACDE"

    expected = _mutation_pool(hidden, attention, sequences, consensus)
    actual = _mutation_context_topk_pool(
        hidden, attention, sequences, consensus, top_k=0
    )

    torch.testing.assert_close(actual, expected)


def test_static_delta_topk_zero_matches_mutation_delta_pool():
    hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    consensus_hidden = torch.ones(1, 6, 3)
    attention = torch.ones(2, 6, dtype=torch.long)
    sequences = ["ACWE", "ACDE"]
    consensus = "ACDE"

    expected = _mutation_pool(
        hidden, attention, sequences, consensus, cons_hidden=consensus_hidden
    )
    actual = _mutation_context_topk_pool(
        hidden,
        attention,
        sequences,
        consensus,
        cons_hidden=consensus_hidden,
        top_k=0,
    )

    torch.testing.assert_close(actual, expected)


def test_trainable_topk_zero_matches_token_mutation_pool():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    input_ids = torch.tensor([[0, 1, 9, 3, 4], [0, 1, 2, 3, 4]])
    consensus_ids = torch.tensor([0, 1, 2, 3, 4])
    attention = torch.ones(2, 5, dtype=torch.long)

    expected = token_mutation_pool(hidden, input_ids, consensus_ids, attention)
    actual = token_mutation_context_topk_pool(
        hidden, input_ids, consensus_ids, attention, top_k=0
    )

    torch.testing.assert_close(actual, expected)


def test_trainable_context_pool_propagates_gradients():
    hidden = torch.randn(1, 6, 4, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 9, 3, 4, 5]])
    consensus_ids = torch.tensor([0, 1, 2, 3, 4, 5])
    attention = torch.ones(1, 6, dtype=torch.long)

    pooled = token_mutation_context_topk_pool(
        hidden,
        input_ids,
        consensus_ids,
        attention,
        top_k=2,
        locality=0.5,
        temperature=1.0,
    )
    pooled.sum().backward()

    assert pooled.shape == (1, 4)
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad) > 0
