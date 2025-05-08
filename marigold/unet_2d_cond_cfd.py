import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class UNet2DConditionModelCFD(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom layers or modifications here
        print("using the customize model.")
        
    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Implement custom forward logic if needed
        return super().forward(sample, timestep, encoder_hidden_states, **kwargs)
