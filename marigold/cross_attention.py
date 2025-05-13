import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
      
      
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=16, input_dim=4, num_heads=4):
        """
        Fusion module that integrates rgb, flow, and depth latents via cross-attention.
        Args:
          embed_dim (int): Dimension to project each latent's channels to for attention.
          num_heads (int): Number of attention heads (embed_dim should be divisible by this).
        """
        super(CrossAttentionFusion, self).__init__()
        # Linear projections to increase channel dimensions for attention (1x1 convolutions).
        self.proj_rgb = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        self.norm_rgb = nn.LayerNorm(embed_dim)
        
        self.proj_depth = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        self.norm_depth = nn.LayerNorm(embed_dim)
        
        # Multi-head cross-attention: depth as query, rgb+flow as key/value.
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Projection to reduce fused depth features back to 4 channels.
        self.mlp = Mlp(in_features=embed_dim)
        self.norm_rgb2 = nn.LayerNorm(embed_dim)
        self.proj_fuse = nn.Linear(embed_dim, input_dim)
        self.new_conv_in = nn.Conv2d(16, 320, kernel_size=3, padding=1)
        
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
        # ---------------------- Cross Attention Part ----------------------
        # 1. Project each input from 4 channels to the embed_dim for attention.
        rgb_feat = self.proj_rgb(rgb_latent)              # [B, embed_dim, H, W]
        rgb_seq  = rgb_feat.flatten(2).transpose(1, 2)    # [B, H*W, embed_dim]
        rgb_q    = self.norm_rgb(rgb_seq)                 # [B, H*W, embed_dim]
        
        depth_feat = self.proj_depth(depth_latent)          # [B, embed_dim, H, W]
        depth_seq  = depth_feat.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        depth_kv   = self.norm_depth(depth_seq)             # [B, H*W, embed_dim]
        
        # 3. Compute cross-attention: Depth features attend to combined RGB+Flow features.
        attn_output, _ = self.cross_attn(query=depth_kv, key=rgb_q, value=rgb_q) # [B, H*W, embed_dim]
        
        x_fused = attn_output + rgb_seq   # [B, H*W, embed_dim]
        
        # ---------------------- MLP Part ---------------------- 
        fused = x_fused + self.mlp(self.norm_rgb2(x_fused)) # [B, H*W, embed_dim]
        # 5. Project fused depth features back to 4 channels.
        fused =  self.proj_fuse(fused)  # [B, H*W, 4]
        # 4. Reshape output back to [B, embed_dim, H, W].
        fused = fused.transpose(1, 2).view(B, -1, H, W)  # [B, 4, H, W]
    
        # Concatenate the original features (residual connection) to preserve original information.
        x = torch.cat((rgb_latent, flow_latent, fused, depth_latent), dim=1) # [B, 16, H, W]
        x = self.new_conv_in(x)
        
        return x
