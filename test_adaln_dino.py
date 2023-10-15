import pytest
import torch
from dinov2.models.vision_transformer import adaLN_vit_base


def test_number_of_parameters():
    def count_parameters(model):
        params = list(model.parameters())
        num_params = sum(p.numel() for p in params)
        return num_params

    # Create models
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda().eval()
    adaLN_dino = adaLN_vit_base().cuda().eval()

    dino_n_params = count_parameters(dino)
    adaLN_dino_n_params = count_parameters(adaLN_dino)
    adaLN_n_params = 0
    for block in adaLN_dino.blocks:
        adaLN_n_params += count_parameters(block.adaLN_modulation)
    assert dino_n_params + adaLN_n_params == adaLN_dino_n_params, "Number of parameters should be the same"

def test_forward_pass():
    # Create models
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda().eval()
    adaLN_dino = adaLN_vit_base().cuda().eval()

    # Forward pass
    x_image = torch.randn(2, 3, 224, 224).cuda()
    x_camera = torch.randn(2, 768).cuda()
    dino_features = dino.forward_features(x_image)
    adaLN_dino_features = adaLN_dino.forward_features(x_image, c=x_camera)

    keys = ['x_norm_clstoken', 'x_norm_patchtokens']
    for k in keys:
        assert torch.allclose(dino_features[k], adaLN_dino_features[k], atol=1e-3), f"Features {k} should be the same"
