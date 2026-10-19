import pytest
import torch


@pytest.mark.parametrize(
    ("head_name", "head_config", "expected_params"),
    [
        ("mean_pool_linear_mil", {}, 1026),
        ("mean_pool_mlp_mil", {"hidden_dim": 64, "dropout": 0.25}, 32962),
        ("max_pool_linear_mil", {}, 1026),
    ],
)
def test_pooling_mil_heads(head_name, head_config, expected_params):
    from heads import HEAD_REGISTRY

    model = HEAD_REGISTRY[head_name](input_dim=512, n_classes=2, **head_config)
    logits, probabilities, prediction = model(torch.randn(17, 512))

    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1, 1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params


def test_pooling_mil_rejects_non_bag_tensor():
    from heads import MeanPoolLinearMIL

    with pytest.raises(ValueError, match=r"expects \[N,D\]"):
        MeanPoolLinearMIL(input_dim=8)(torch.randn(1, 3, 8))
