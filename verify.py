import numpy as np
import timm
import torch
import torch.nn as nn
from vit import VisionTransformer


def get_n_params(
    module: nn.Module
):
    return sum(
        p.numel() for p in module.parameters()
        if p.requires_grad
    )


def assert_tensors_equal(
    t1: torch.FloatTensor,
    t2: torch.FloatTensor,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> bool:
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    return np.testing.assert_allclose(a1, a2, rtol=rtol, atol=atol)


model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
    "img_size": 384,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()

for (n_o, p_o), (n_c, p_c) in zip(
    model_official.named_parameters(), model_custom.named_parameters()
):
    assert p_o.numel() == p_c.numel()

    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data)


inp = torch.rand(1, 3, 384, 384)
res_o = model_official(inp)
res_c = model_custom(inp)

assert get_n_params(model_custom) == get_n_params(model_official)
assert_tensors_equal(res_o, res_c)

torch.save(model_custom, "model.pth")
