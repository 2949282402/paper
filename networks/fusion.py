"""
Dual-Branch Fusion Module for Video Deepfake Detection

Combines RGB and optical flow features (256-dim each) for final classification.
Supports multiple fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """
    Feature fusion module for dual-branch architecture.

    Fusion strategies:
    - concat: Concatenate features [256+256=512] -> MLP -> 1
    - add: Element-wise addition -> MLP -> 1
    - bilinear: Bilinear pooling for feature interaction
    - attention: Cross-attention fusion
    """

    def __init__(self, feature_dim=256, fusion_type='concat', hidden_dim=128, dropout=0.5):
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim

        if fusion_type == 'concat':
            # Concatenation fusion: [rgb; optical] -> fc -> output
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == 'add':
            # Addition fusion: rgb + optical -> fc -> output
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == 'bilinear':
            # Bilinear fusion: outer product with low-rank approximation
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, hidden_dim)
            self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == 'attention':
            # Cross-attention fusion
            self.query = nn.Linear(feature_dim, feature_dim)
            self.key = nn.Linear(feature_dim, feature_dim)
            self.value = nn.Linear(feature_dim, feature_dim)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == 'gated':
            # Gated fusion: learn adaptive weights for each branch
            # Use single FC + Softmax for stable gradients
            self.gate = nn.Linear(feature_dim * 2, 2)  # Output 2 logits for [rgb, optical]
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_rgb, feat_optical):
        """
        Args:
            feat_rgb: RGB branch features [batch, 256]
            feat_optical: Optical flow branch features [batch, 256]

        Returns:
            logits: Classification logits [batch, 1]
        """
        if self.fusion_type == 'concat':
            fused = torch.cat([feat_rgb, feat_optical], dim=1)  # [batch, 512]
            out = self.classifier(fused)

        elif self.fusion_type == 'add':
            fused = feat_rgb + feat_optical  # [batch, 256]
            out = self.classifier(fused)

        elif self.fusion_type == 'bilinear':
            fused = self.bilinear(feat_rgb, feat_optical)  # [batch, hidden_dim]
            out = self.classifier(fused)

        elif self.fusion_type == 'attention':
            # Cross-attention: RGB attends to optical, and vice versa
            q_rgb = self.query(feat_rgb)
            k_optical = self.key(feat_optical)
            v_optical = self.value(feat_optical)

            # Attention weight
            attn = torch.sum(q_rgb * k_optical, dim=1, keepdim=True) / (self.feature_dim ** 0.5)
            attn = torch.sigmoid(attn)

            # Attended features
            attended = attn * v_optical + (1 - attn) * feat_rgb
            fused = torch.cat([attended, feat_optical], dim=1)
            out = self.classifier(fused)

        elif self.fusion_type == 'gated':
            # Gated fusion with Softmax (stable gradients, naturally normalized)
            concat = torch.cat([feat_rgb, feat_optical], dim=1)  # [B, 512]
            gate_logits = self.gate(concat)  # [B, 2]
            gates = torch.softmax(gate_logits, dim=1)  # [B, 2], sum=1
            gate_r = gates[:, 0:1]  # [B, 1]
            gate_o = gates[:, 1:2]  # [B, 1]
            fused = gate_r * feat_rgb + gate_o * feat_optical  # [B, 256]
            out = self.classifier(fused)

        return out


class DualBranchNet(nn.Module):
    """
    Complete dual-branch network for video deepfake detection.

    Architecture:
        RGB Image -> RGB Branch (ResNet/FreqNet) -> 256-dim features
        Optical Flow -> Optical Branch (ResNet/FreqNet) -> 256-dim features
        [RGB features, Optical features] -> Fusion Module -> Classification
    """

    def __init__(self, rgb_branch, optical_branch, fusion_type='concat',
                 feature_dim=256, hidden_dim=128, dropout=0.5):
        super(DualBranchNet, self).__init__()
        self.rgb_branch = rgb_branch
        self.optical_branch = optical_branch
        self.fusion = FusionModule(
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, rgb_input, optical_input):
        """
        Args:
            rgb_input: RGB images [batch, 3, H, W]
            optical_input: Optical flow images [batch, 3, H, W]

        Returns:
            logits: Classification logits [batch, 1]
        """
        # Extract features from both branches
        feat_rgb = self.rgb_branch(rgb_input)  # [batch, 256]
        feat_optical = self.optical_branch(optical_input)  # [batch, 256]

        # Fuse features and classify
        logits = self.fusion(feat_rgb, feat_optical)

        return logits

    def get_features(self, rgb_input, optical_input):
        """
        Get intermediate features for analysis.

        Returns:
            feat_rgb, feat_optical, fused_logits
        """
        feat_rgb = self.rgb_branch(rgb_input)
        feat_optical = self.optical_branch(optical_input)
        logits = self.fusion(feat_rgb, feat_optical)
        return feat_rgb, feat_optical, logits


def create_dual_branch_model(arch_rgb='freqnet', arch_optical='resnet50',
                              fusion_type='concat', feature_dim=256,
                              pretrained=True):
    """
    Factory function to create dual-branch model.

    Args:
        arch_rgb: Architecture for RGB branch ('freqnet' or 'resnetXX')
        arch_optical: Architecture for optical flow branch
        fusion_type: Fusion strategy ('concat', 'add', 'bilinear', 'attention', 'gated')
        feature_dim: Feature dimension for each branch (default 256)
        pretrained: Whether to use pretrained weights for ResNet

    Returns:
        DualBranchNet model

    Example:
        model = create_dual_branch_model(
            arch_rgb='freqnet',
            arch_optical='resnet50',
            fusion_type='concat'
        )
        logits = model(rgb_images, optical_images)
    """
    # Create RGB branch
    if 'freqnet' in arch_rgb:
        from networks.freqnet import freqnet
        rgb_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        rgb_branch = resnet_fn.get(arch_rgb, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create optical flow branch
    if 'freqnet' in arch_optical:
        from networks.freqnet import freqnet
        optical_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        optical_branch = resnet_fn.get(arch_optical, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create dual-branch model
    model = DualBranchNet(
        rgb_branch=rgb_branch,
        optical_branch=optical_branch,
        fusion_type=fusion_type,
        feature_dim=feature_dim
    )

    return model


# ============ Transformer Temporal Aggregation Module ============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal aggregation module.

    Takes a sequence of frame features and outputs a single video-level feature.
    """

    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.feature_dim = feature_dim

        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: Frame features [batch, num_frames, feature_dim]

        Returns:
            Video-level feature [batch, feature_dim]
        """
        batch_size = x.size(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, feature_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_frames+1, feature_dim]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Extract CLS token output as video-level feature
        x = x[:, 0, :]  # [batch, feature_dim]
        x = self.norm(x)

        return x


class VideoFusionModule(nn.Module):
    """
    Video-level fusion module combining RGB and optical flow temporal features.

    Architecture:
        RGB frames -> RGB Branch -> [T, 256] -> Temporal Transformer -> 256
        Optical frames -> Optical Branch -> [T, 256] -> Temporal Transformer -> 256
        [RGB video feat, Optical video feat] -> Fusion -> Classification
    """

    def __init__(self, feature_dim=256, fusion_type='concat', hidden_dim=128,
                 num_heads=8, num_layers=2, dropout=0.5):
        super(VideoFusionModule, self).__init__()

        # Temporal aggregation for each branch
        self.rgb_temporal = TemporalTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.optical_temporal = TemporalTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Feature fusion (same as frame-level fusion)
        self.fusion = FusionModule(
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, rgb_features, optical_features):
        """
        Args:
            rgb_features: [batch, num_frames, feature_dim]
            optical_features: [batch, num_frames, feature_dim]

        Returns:
            logits: [batch, 1]
        """
        # Temporal aggregation
        rgb_video_feat = self.rgb_temporal(rgb_features)  # [batch, feature_dim]
        optical_video_feat = self.optical_temporal(optical_features)  # [batch, feature_dim]

        # Fusion and classification
        logits = self.fusion(rgb_video_feat, optical_video_feat)

        return logits


class VideoDualBranchNet(nn.Module):
    """
    Video-level dual-branch network with Transformer temporal aggregation.

    Architecture:
        RGB Video [B, T, 3, H, W] -> RGB Branch -> [B, T, 256] -> Temporal Transformer -> [B, 256]
        Optical Video [B, T, 3, H, W] -> Optical Branch -> [B, T, 256] -> Temporal Transformer -> [B, 256]
        [RGB feat, Optical feat] -> Fusion -> Video Classification
    """

    def __init__(self, rgb_branch, optical_branch, fusion_type='concat',
                 feature_dim=256, hidden_dim=128, num_heads=8, num_layers=2, dropout=0.5):
        super(VideoDualBranchNet, self).__init__()
        self.rgb_branch = rgb_branch
        self.optical_branch = optical_branch

        # Video-level fusion module with temporal transformers
        self.video_fusion = VideoFusionModule(
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, rgb_video, optical_video):
        """
        Args:
            rgb_video: [batch, num_frames, 3, H_rgb, W_rgb]
            optical_video: [batch, num_frames, 3, H_optical, W_optical]

        Returns:
            logits: [batch, 1]
        """
        batch_size, num_frames = rgb_video.shape[:2]

        # Reshape for batch processing: [B*T, 3, H, W]
        rgb_flat = rgb_video.view(batch_size * num_frames, *rgb_video.shape[2:])
        optical_flat = optical_video.view(batch_size * num_frames, *optical_video.shape[2:])

        # Extract frame-level features
        rgb_features = self.rgb_branch(rgb_flat)  # [B*T, feature_dim]
        optical_features = self.optical_branch(optical_flat)  # [B*T, feature_dim]

        # Reshape back to sequence: [B, T, feature_dim]
        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        optical_features = optical_features.view(batch_size, num_frames, -1)

        # Video-level fusion with temporal aggregation
        logits = self.video_fusion(rgb_features, optical_features)

        return logits

    def get_features(self, rgb_video, optical_video):
        """Get intermediate features for analysis."""
        batch_size, num_frames = rgb_video.shape[:2]

        rgb_flat = rgb_video.view(batch_size * num_frames, *rgb_video.shape[2:])
        optical_flat = optical_video.view(batch_size * num_frames, *optical_video.shape[2:])

        rgb_features = self.rgb_branch(rgb_flat).view(batch_size, num_frames, -1)
        optical_features = self.optical_branch(optical_flat).view(batch_size, num_frames, -1)

        rgb_video_feat = self.video_fusion.rgb_temporal(rgb_features)
        optical_video_feat = self.video_fusion.optical_temporal(optical_features)

        return rgb_video_feat, optical_video_feat


def create_video_dual_branch_model(arch_rgb='freqnet', arch_optical='resnet50',
                                    fusion_type='concat', feature_dim=256,
                                    num_heads=8, num_layers=2, pretrained=True):
    """
    Factory function to create video-level dual-branch model with Transformer.

    Args:
        arch_rgb: Architecture for RGB branch ('freqnet' or 'resnetXX')
        arch_optical: Architecture for optical flow branch
        fusion_type: Fusion strategy ('concat', 'add', 'bilinear', 'attention', 'gated')
        feature_dim: Feature dimension (default 256)
        num_heads: Number of attention heads in Transformer
        num_layers: Number of Transformer encoder layers
        pretrained: Whether to use pretrained weights for ResNet

    Returns:
        VideoDualBranchNet model

    Example:
        model = create_video_dual_branch_model(
            arch_rgb='freqnet',
            arch_optical='resnet50',
            fusion_type='concat',
            num_frames=16
        )
        # rgb_video: [batch, 16, 3, 224, 224]
        # optical_video: [batch, 16, 3, 448, 448]
        logits = model(rgb_video, optical_video)
    """
    # Create RGB branch
    if 'freqnet' in arch_rgb:
        from networks.freqnet import freqnet
        rgb_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        rgb_branch = resnet_fn.get(arch_rgb, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create optical flow branch
    if 'freqnet' in arch_optical:
        from networks.freqnet import freqnet
        optical_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        optical_branch = resnet_fn.get(arch_optical, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create video-level dual-branch model
    model = VideoDualBranchNet(
        rgb_branch=rgb_branch,
        optical_branch=optical_branch,
        fusion_type=fusion_type,
        feature_dim=feature_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    return model


# ============ Early Fusion: Frame-Level Fusion + Temporal Transformer ============

class FrameLevelFusionModule(nn.Module):
    """
    Frame-level fusion module that combines RGB and optical flow features.

    Unlike FusionModule which outputs classification logits,
    this module outputs fused features for subsequent temporal processing.

    Fusion strategies:
    - concat: [RGB; Optical] -> Linear -> fused_dim
    - add: RGB + Optical (requires same dim)
    - bilinear: Bilinear pooling
    - attention: Cross-attention fusion
    - gated: Learned gating
    """

    def __init__(self, feature_dim=256, fusion_type='concat', fused_dim=512, dropout=0.1):
        super(FrameLevelFusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim
        self.fused_dim = fused_dim

        if fusion_type == 'concat':
            # Concatenation: [256, 256] -> 512 -> fused_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * 2, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'add':
            # Addition: element-wise add -> project to fused_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'bilinear':
            # Bilinear fusion
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, fused_dim)
            self.fusion_layer = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'attention':
            # Cross-modal attention fusion
            self.query_rgb = nn.Linear(feature_dim, feature_dim)
            self.key_optical = nn.Linear(feature_dim, feature_dim)
            self.value_optical = nn.Linear(feature_dim, feature_dim)
            self.query_optical = nn.Linear(feature_dim, feature_dim)
            self.key_rgb = nn.Linear(feature_dim, feature_dim)
            self.value_rgb = nn.Linear(feature_dim, feature_dim)
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * 2, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'gated':
            # Gated fusion with learned weights
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_rgb, feat_optical):
        """
        Args:
            feat_rgb: [batch, 256] or [batch, T, 256]
            feat_optical: [batch, 256] or [batch, T, 256]

        Returns:
            fused: [batch, fused_dim] or [batch, T, fused_dim]
        """
        if self.fusion_type == 'concat':
            concat = torch.cat([feat_rgb, feat_optical], dim=-1)
            fused = self.fusion_layer(concat)

        elif self.fusion_type == 'add':
            added = feat_rgb + feat_optical
            fused = self.fusion_layer(added)

        elif self.fusion_type == 'bilinear':
            # nn.Bilinear only supports 2D input, need to handle 3D case
            if feat_rgb.dim() == 3:
                # [batch, T, feature_dim] -> [batch*T, feature_dim]
                batch_size, seq_len, feat_dim = feat_rgb.shape
                feat_rgb_2d = feat_rgb.view(-1, feat_dim)
                feat_optical_2d = feat_optical.view(-1, feat_optical.shape[-1])
                bilinear_out = self.bilinear(feat_rgb_2d, feat_optical_2d)
                # [batch*T, fused_dim] -> [batch, T, fused_dim]
                bilinear_out = bilinear_out.view(batch_size, seq_len, -1)
            else:
                bilinear_out = self.bilinear(feat_rgb, feat_optical)
            fused = self.fusion_layer(bilinear_out)

        elif self.fusion_type == 'attention':
            # RGB attends to Optical
            q_rgb = self.query_rgb(feat_rgb)
            k_opt = self.key_optical(feat_optical)
            v_opt = self.value_optical(feat_optical)
            attn_rgb = torch.sigmoid(torch.sum(q_rgb * k_opt, dim=-1, keepdim=True) / (self.feature_dim ** 0.5))
            attended_rgb = attn_rgb * v_opt + (1 - attn_rgb) * feat_rgb

            # Optical attends to RGB
            q_opt = self.query_optical(feat_optical)
            k_rgb = self.key_rgb(feat_rgb)
            v_rgb = self.value_rgb(feat_rgb)
            attn_opt = torch.sigmoid(torch.sum(q_opt * k_rgb, dim=-1, keepdim=True) / (self.feature_dim ** 0.5))
            attended_opt = attn_opt * v_rgb + (1 - attn_opt) * feat_optical

            concat = torch.cat([attended_rgb, attended_opt], dim=-1)
            fused = self.fusion_layer(concat)

        elif self.fusion_type == 'gated':
            concat = torch.cat([feat_rgb, feat_optical], dim=-1)
            gates = self.gate(concat)  # [batch, 2] or [batch, T, 2]
            weighted = gates[..., 0:1] * feat_rgb + gates[..., 1:2] * feat_optical
            fused = self.fusion_layer(weighted)

        return fused


class EarlyFusionVideoModule(nn.Module):
    """
    Early Fusion module: Frame-level fusion followed by Temporal Transformer.

    Architecture:
        RGB [T, 256] + Optical [T, 256] -> Frame Fusion -> [T, 512] -> Transformer -> Classification

    This captures cross-modal interactions at frame level before temporal aggregation.
    """

    def __init__(self, feature_dim=256, fused_dim=512, fusion_type='concat',
                 num_heads=8, num_layers=2, hidden_dim=128, dropout=0.1):
        super(EarlyFusionVideoModule, self).__init__()

        # Frame-level fusion
        self.frame_fusion = FrameLevelFusionModule(
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            fused_dim=fused_dim,
            dropout=dropout
        )

        # Single Transformer for temporal aggregation on fused features
        self.temporal_transformer = TemporalTransformer(
            feature_dim=fused_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, rgb_features, optical_features):
        """
        Args:
            rgb_features: [batch, num_frames, feature_dim]
            optical_features: [batch, num_frames, feature_dim]

        Returns:
            logits: [batch, 1]
        """
        # Frame-level fusion: [batch, T, 256] + [batch, T, 256] -> [batch, T, 512]
        fused_features = self.frame_fusion(rgb_features, optical_features)

        # Temporal aggregation: [batch, T, 512] -> [batch, 512]
        video_feature = self.temporal_transformer(fused_features)

        # Classification
        logits = self.classifier(video_feature)

        return logits

    def get_fused_features(self, rgb_features, optical_features):
        """Get intermediate fused features for analysis."""
        fused_features = self.frame_fusion(rgb_features, optical_features)
        video_feature = self.temporal_transformer(fused_features)
        return fused_features, video_feature


class EarlyFusionVideoNet(nn.Module):
    """
    Video-level network with Early Fusion strategy.

    Architecture (Early Fusion):
        RGB Video [B, T, 3, H, W] -> RGB Branch -> [B, T, 256]
                                                           ↘
                                                             Frame Fusion -> [B, T, 512] -> Transformer -> Classification
                                                           ↗
        Optical Video [B, T, 3, H, W] -> Optical Branch -> [B, T, 256]

    Compared to Late Fusion (VideoDualBranchNet):
    - Early Fusion: Fuse first, then temporal aggregate
    - Late Fusion: Temporal aggregate separately, then fuse

    Early Fusion is better for capturing cross-modal inconsistencies at frame level,
    which is crucial for deepfake detection.
    """

    def __init__(self, rgb_branch, optical_branch, fusion_type='concat',
                 feature_dim=256, fused_dim=512, hidden_dim=128,
                 num_heads=8, num_layers=2, dropout=0.1):
        super(EarlyFusionVideoNet, self).__init__()
        self.rgb_branch = rgb_branch
        self.optical_branch = optical_branch

        # Early fusion module
        self.early_fusion = EarlyFusionVideoModule(
            feature_dim=feature_dim,
            fused_dim=fused_dim,
            fusion_type=fusion_type,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, rgb_video, optical_video):
        """
        Args:
            rgb_video: [batch, num_frames, 3, H_rgb, W_rgb]
            optical_video: [batch, num_frames, 3, H_optical, W_optical]

        Returns:
            logits: [batch, 1]
        """
        batch_size, num_frames = rgb_video.shape[:2]

        # Reshape for batch processing: [B*T, 3, H, W]
        rgb_flat = rgb_video.view(batch_size * num_frames, *rgb_video.shape[2:])
        optical_flat = optical_video.view(batch_size * num_frames, *optical_video.shape[2:])

        # Extract frame-level features
        rgb_features = self.rgb_branch(rgb_flat)  # [B*T, feature_dim]
        optical_features = self.optical_branch(optical_flat)  # [B*T, feature_dim]

        # Reshape back to sequence: [B, T, feature_dim]
        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        optical_features = optical_features.view(batch_size, num_frames, -1)

        # Early fusion and classification
        logits = self.early_fusion(rgb_features, optical_features)

        return logits

    def get_features(self, rgb_video, optical_video):
        """Get intermediate features for analysis."""
        batch_size, num_frames = rgb_video.shape[:2]

        rgb_flat = rgb_video.view(batch_size * num_frames, *rgb_video.shape[2:])
        optical_flat = optical_video.view(batch_size * num_frames, *optical_video.shape[2:])

        rgb_features = self.rgb_branch(rgb_flat).view(batch_size, num_frames, -1)
        optical_features = self.optical_branch(optical_flat).view(batch_size, num_frames, -1)

        fused_features, video_feature = self.early_fusion.get_fused_features(rgb_features, optical_features)

        return rgb_features, optical_features, fused_features, video_feature


def create_early_fusion_video_model(arch_rgb='freqnet', arch_optical='resnet50',
                                     fusion_type='concat', feature_dim=256, fused_dim=512,
                                     num_heads=8, num_layers=2, pretrained=True):
    """
    Factory function to create video-level model with Early Fusion.

    Early Fusion Architecture:
        RGB frame -> 256 ↘
                          Frame Fusion -> 512 -> [T, 512] -> Transformer -> Classification
        Optical frame -> 256 ↗

    Args:
        arch_rgb: Architecture for RGB branch ('freqnet' or 'resnetXX')
        arch_optical: Architecture for optical flow branch
        fusion_type: Fusion strategy ('concat', 'add', 'bilinear', 'attention', 'gated')
        feature_dim: Feature dimension for each branch (default 256)
        fused_dim: Fused feature dimension (default 512)
        num_heads: Number of attention heads in Transformer
        num_layers: Number of Transformer encoder layers
        pretrained: Whether to use pretrained weights for ResNet

    Returns:
        EarlyFusionVideoNet model

    Example:
        model = create_early_fusion_video_model(
            arch_rgb='freqnet',
            arch_optical='resnet50',
            fusion_type='attention',  # Cross-modal attention for better interaction
            fused_dim=512
        )
        # rgb_video: [batch, 16, 3, 224, 224]
        # optical_video: [batch, 16, 3, 448, 448]
        logits = model(rgb_video, optical_video)
    """
    # Create RGB branch
    if 'freqnet' in arch_rgb:
        from networks.freqnet import freqnet
        rgb_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        rgb_branch = resnet_fn.get(arch_rgb, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create optical flow branch
    if 'freqnet' in arch_optical:
        from networks.freqnet import freqnet
        optical_branch = freqnet(return_feature=True, feature_dim=feature_dim)
    else:
        from networks.resnet import resnet50, resnet18, resnet34
        resnet_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        optical_branch = resnet_fn.get(arch_optical, resnet50)(
            pretrained=pretrained,
            return_feature=True,
            feature_dim=feature_dim
        )

    # Create early fusion video model
    model = EarlyFusionVideoNet(
        rgb_branch=rgb_branch,
        optical_branch=optical_branch,
        fusion_type=fusion_type,
        feature_dim=feature_dim,
        fused_dim=fused_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )

    return model
