import torch
import torch.nn as nn



class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4):
        """
        Fusion module that integrates rgb, flow, and depth latents via cross-attention.
        Args:
          embed_dim (int): Dimension to project each latent's channels to for attention.
          num_heads (int): Number of attention heads (embed_dim should be divisible by this).
        """
        super(CrossAttentionFusion, self).__init__()
        # Linear projections to increase channel dimensions for attention (1x1 convolutions).
        self.proj_rgb   = nn.Conv2d(4, embed_dim, kernel_size=1)
        self.proj_flow  = nn.Conv2d(4, embed_dim, kernel_size=1)
        self.proj_depth = nn.Conv2d(4, embed_dim, kernel_size=1)
        # Multi-head cross-attention: depth as query, rgb+flow as key/value.
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Projection to reduce fused depth features back to 4 channels.
        self.proj_out_depth = nn.Conv2d(embed_dim * 3, 4, kernel_size=1)
        
    def forward(self, sample):
        """
        Forward pass of the fusion module.
        Inputs:
          sample: tensor of shape [B, 12, H, W]
          split as (rgb_latent, flow_latent, depth_latent): tensors of shape [B, 4, H, W].
        Output:
          fused: tensor of shape [B, 12, H, W] combining all modalities with attention applied.
        """
        rgb_latent, flow_latent, depth_latent = torch.split(sample, [4, 4, 4], dim=1)
        B, _, H, W = depth_latent.shape
        # 1. Project each input from 4 channels to the embed_dim for attention.
        rgb_feat   = self.proj_rgb(rgb_latent)    # [B, embed_dim, H, W]
        flow_feat  = self.proj_flow(flow_latent)  # [B, embed_dim, H, W]
        depth_feat = self.proj_depth(depth_latent)  # [B, embed_dim, H, W]
        
        # 2. Flatten spatial dimensions to sequence (batch_first = True -> shape [B, seq_len, embed_dim]).
        # Here, seq_len = H*W for each latent.
        rgb_seq   = rgb_feat.flatten(2).transpose(1, 2)    # [B, H*W, embed_dim]
        flow_seq  = flow_feat.flatten(2).transpose(1, 2)   # [B, H*W, embed_dim]
        depth_seq = depth_feat.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 3. Compute cross-attention: Depth features attend to combined RGB+Flow features.
        attn_output1, _ = self.cross_attn1(query=depth_seq, key=flow_seq, value=flow_seq) # [B, H*W, embed_dim]
        # 4. Reshape attention output back to [B, embed_dim, H, W].
        attn_output1 = attn_output1.transpose(1, 2).view(B, -1, H, W)  # [B, embed_dim, H, W]
        
        attn_output2, _ = self.cross_attn1(query=depth_seq, key=rgb_seq, value=rgb_seq) # [B, H*W, embed_dim]
        attn_output2 = attn_output2.transpose(1, 2).view(B, -1, H, W)  # [B, embed_dim, H, W]
        
        # Add the original depth features (residual connection) to preserve original information.
        fused_depth_feat = torch.cat((depth_feat, attn_output1, attn_output2), dim=1)
        
        # 5. Project fused depth features back to 4 channels.
        fused = self.proj_out_depth(fused_depth_feat) + depth_latent  # [B, 4, H, W]
        
        # 6. Concatenate rgb, flow, and fused depth back along channel dimension to get 12-channel output.
        # fused = torch.cat([rgb_latent, flow_latent, fused_depth], dim=1)  # [B, 12, H, W]
        return fused
