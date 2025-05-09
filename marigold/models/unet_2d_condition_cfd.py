from diffusers import UNet2DConditionModel
from marigold.models.cross_attention import CrossAttentionFusion


class UNet2DConditionModelCFD(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_block = CrossAttentionFusion(embed_dim=32, num_heads=4)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Implement custom forward logic if needed
        sample = self.fusion_block(sample)
        return super().forward(sample, timestep, encoder_hidden_states, **kwargs)
