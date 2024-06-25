import re
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

from typing import Optional

class ViT(torch.nn.Module):
    """
    ViT model with configurable size, patch size, pre-trained weights, and trainable layers.
    """
    def __init__(
        self, 
        size: str, 
        patch_size: int, 
        num_classes = int, 
        image_size: int = None, 
        weigths: str = 'imagenet',
        pretrained: bool = False, 
        trainable_layers: Optional[int] = None):
        """
        Initializes the ViT model with the specified parameters.

        Args:
            size: String specifying the model size (base, large, or huge).
            patch_size: Integer representing the patch size for image processing.
            num_classes (int): Number of classes the model will be trained to classify.
            image_size (int, optional): Integer representing the input image size. Defaults to None (use default size based on model size).
            weights (str, optional): String indicating the pre-trained weights to load ('imagenet' or 'custom'). Defaults to 'imagenet'.
            pretrained: Boolean indicating whether to load pre-trained weights.
            trainable_layers: Integer specifying the number of final layers to fine-tune (optional).
        """
        super(ViT, self).__init__()

        self.image_size = image_size
        self.pretrained = pretrained
        self.trainable_layers = trainable_layers

        # Validate size and patch_size combination
        valid_sizes = {"base": [16, 32], "large": [16, 32], "huge": [14]}
        if size not in valid_sizes or patch_size not in valid_sizes[size]:
            raise ValueError(f"Invalid combination of size '{size}' and patch_size {patch_size}.")

        weights = 'DEFAULT' if self.pretrained and weigths=='imagenet' else None

        # Load appropriate model based on size and patch size
        model_func = {
            ("base", 16): models.vit_b_16,
            ("base", 32): models.vit_b_32,
            ("large", 16): models.vit_l_16,
            ("large", 32): models.vit_l_32,
            ("huge", 14): models.vit_h_14,
        }
        if self.image_size: 
            self.model = model_func[(size, patch_size)](weights=weights, image_size=self.image_size)
        else:
            self.model = model_func[(size, patch_size)](weights=weights)
        self.model_name = model_func[(size, patch_size)].__name__
        
        if pretrained:
            # Freeze all layers by default
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze some layers if trainable_layers is specified
            if self.trainable_layers is not None:
                self._unfreeze_layers(self.trainable_layers)

        # Update head layer for classification
        self.num_classes = num_classes
        num_features_in = self.model.heads.head.in_features
        self.model.heads.head = torch.nn.Linear(num_features_in, self.num_classes)

    def _unfreeze_layers(self, trainable_layers: int):
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

def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False) -> "OrderedDict[str, torch.Tensor]":

    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["model.encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["model.encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
    
if __name__ == '__main__':
    model = ViT(size='base', patch_size=16, image_size=768, num_classes=2, pretrained=True, trainable_layers=6)