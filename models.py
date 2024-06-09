import torch
from torchvision import models

from typing import Optional

def vit(size: str, patch_size: int = 16, pretrained: bool = False, trainable_layers: Optional[int] = 11) -> torch.nn.Module:
    """
    Creates a ViT model with specified size, patch size, pre-trained weights, and trainable layers.

    Args:
        size: String specifying the model size (base, large, or huge).
        patch_size: Integer representing the patch size for image processing.
        pretrained: Boolean indicating whether to load pre-trained weights.
        trainable_layers: Integer specifying the number of final layers to fine-tune (optional).

    Returns:
        A PyTorch model instance of the specified ViT architecture.
    """

    # Valid size and patch_size combination
    valid_sizes = {"base": [16, 32], "large": [16, 32], "huge": [14]}
    if size not in valid_sizes or patch_size not in valid_sizes[size]:
        raise ValueError(f"Invalid combination of size '{size}' and patch_size {patch_size}.")

    weights = 'DEFAULT' if pretrained else None

    # Load appropriate model based on size and patch size
    model_func = {
        ("base", 16): models.vit_b_16,
        ("base", 32): models.vit_b_32,
        ("large", 16): models.vit_l_16,
        ("large", 32): models.vit_l_32,
        ("huge", 14): models.vit_h_14,
    }[(size, patch_size)]
    model = model_func(weights=weights)

    # Freeze layers if trainable_layers is specified
    if trainable_layers is not None:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.parameters()[-trainable_layers:]:
            param.requires_grad = True

    # Update head layer for classification
    num_classes = len(df.label.unique())
    num_features_in = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_features_in, num_classes)

    return model

# Example usage
model = vit('base', pretrained=True)