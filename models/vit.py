import re

import torch
from torchvision import models

from typing import Optional

class ViT(torch.nn.Module):
    """
    ViT model with configurable size, patch size, pre-trained weights, and trainable layers.
    """
    def __init__(self, size: str, patch_size: int = 16, num_classes = int, pretrained: bool = False, trainable_layers: Optional[int] = None):
        """
        Initializes the ViT model with the specified parameters.

        Args:
            size: String specifying the model size (base, large, or huge).
            patch_size: Integer representing the patch size for image processing.
            pretrained: Boolean indicating whether to load pre-trained weights.
            trainable_layers: Integer specifying the number of final layers to fine-tune (optional).
        """
        super(ViT, self).__init__()

        self.pretrained = pretrained
        self.trainable_layers = trainable_layers

        # Validate size and patch_size combination
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
        }
        self.model = model_func[(size, patch_size)](weights=weights)
        self.model_name = model_func[(size, patch_size)].__name__
        
        if pretrained:
            # Freeze all layers by default
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze some layers if trainable_layers is specified
            if self.trainable_layers is not None:
                self._freeze_layers(self.trainable_layers)

        # Update head layer for classification
        self.num_classes = num_classes
        num_features_in = self.model.heads.head.in_features
        self.model.heads.head = torch.nn.Linear(num_features_in, self.num_classes)

    def _freeze_layers(self, trainable_layers: int):
        """
        Freezes all model layers except the last `trainable_layers`.

        Args:
            trainable_layers: Integer specifying the number of final layers to fine-tune.
        """
        encoder_layers = dict(self.model.encoder.layers.named_children())

        # Find the number of encoder layers 
        num_encoder_layers = len(encoder_layers)

        if trainable_layers > num_encoder_layers:
            raise(ValueError(f'Invalid number for trainable layers. Please use n < {num_encoder_layers}'))
        
        start_layer_index = num_encoder_layers - trainable_layers

        for layer_name, layer in encoder_layers.items():
            match = re.search(r"encoder_layer_(\d+)", layer_name)
            if match and int(match.group(1)) >= start_layer_index:
                for param in layer.parameters():
                    param.requires_grad = True

        for param in self.model.encoder.ln.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)