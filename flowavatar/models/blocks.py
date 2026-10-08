import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    """Residual recurrent block: LayerNorm -> BiGRU (or LSTM) -> projection.

    NOTE: the recurrent layer is stored as ``self.lstm`` for both variants.
    This attribute name is part of the released checkpoints' state_dict keys
    and must not be renamed.
    """
    def __init__(self, hidden_size, use_lstm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm
        self.ln = nn.LayerNorm(hidden_size)

        if use_lstm:
            # Unidirectional LSTM (Unity Sentis has no GRU operator)
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False
            )
            self.projection = nn.Linear(hidden_size, hidden_size)
        else:
            # Bidirectional GRU (default, best accuracy)
            self.lstm = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True
            )
            self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        residual = x
        x = self.ln(x)

        x, _ = self.lstm(x)
        x = self.projection(x)

        return x + residual


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor with balanced importance
    """
    def __init__(self, hidden_size=256, use_lstm=False):
        super().__init__()

        # Activation function based on configuration
        self.activation = nn.ReLU if use_lstm else nn.LeakyReLU

        # Calculate embedding dimensions with appropriate weighting
        self.part_embed_dim = hidden_size // 16  # For body parts

        # Body part embedders (head and hands) - feature index ranges into the
        # 90-D input: (rot_start, rot_end, rot_vel_start, rot_vel_end,
        #              pos_start, pos_end, pos_vel_start, pos_vel_end)
        self.body_parts = {
            'head': (0, 6, 18, 24, 36, 39, 45, 48),
            'lhand': (6, 12, 24, 30, 39, 42, 48, 51),
            'rhand': (12, 18, 30, 36, 42, 45, 51, 54)
        }

        # Head space embedders
        self.head_space_parts = {
            'lhand_inhead': (54, 60, 66, 72, 78, 81, 84, 87),
            'rhand_inhead': (60, 66, 72, 78, 81, 84, 87, 90)
        }

        # Create standard embedders for body parts
        self.rot_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(6, self.part_embed_dim),
                self.activation()
            ) for part in self.body_parts
        })

        self.rot_vel_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(6, self.part_embed_dim),
                self.activation()
            ) for part in self.body_parts
        })

        self.pos_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(3, self.part_embed_dim),
                self.activation()
            ) for part in self.body_parts
        })

        self.pos_vel_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(3, self.part_embed_dim),
                self.activation()
            ) for part in self.body_parts
        })

        # Create embedders for head space features
        self.head_space_rot_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(6, self.part_embed_dim),
                self.activation()
            ) for part in self.head_space_parts
        })

        self.head_space_rot_vel_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(6, self.part_embed_dim),
                self.activation()
            ) for part in self.head_space_parts
        })

        self.head_space_pos_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(3, self.part_embed_dim),
                self.activation()
            ) for part in self.head_space_parts
        })

        self.head_space_pos_vel_embedders = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(3, self.part_embed_dim),
                self.activation()
            ) for part in self.head_space_parts
        })

        # Calculate total input dimension to projection layer
        total_embed_dim = (
            (self.part_embed_dim * 4 * len(self.body_parts)) +
            (self.part_embed_dim * 4 * len(self.head_space_parts))
        )

        self.projection = nn.Linear(total_embed_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Extract features from input tensor
        Args:
            x: Input tensor [B, T, 90] or [B, 90]
        Returns:
            features: Embedded features [B, T, hidden_size] or [B, hidden_size]
        """
        # Handle single frame vs sequence
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        embeddings = []

        # Process each body part
        for part, (rot_s, rot_e, rot_vel_s, rot_vel_e, pos_s, pos_e, pos_vel_s, pos_vel_e) in self.body_parts.items():
            # Get the corresponding embedder for each feature type
            rot_embed = self.rot_embedders[part](x[..., rot_s:rot_e])
            rot_vel_embed = self.rot_vel_embedders[part](x[..., rot_vel_s:rot_vel_e])
            pos_embed = self.pos_embedders[part](x[..., pos_s:pos_e])
            pos_vel_embed = self.pos_vel_embedders[part](x[..., pos_vel_s:pos_vel_e])

            # Add the embedded features to our list
            embeddings.extend([rot_embed, rot_vel_embed, pos_embed, pos_vel_embed])

        # Process head space features
        for part, (rot_s, rot_e, rot_vel_s, rot_vel_e, pos_s, pos_e, pos_vel_s, pos_vel_e) in self.head_space_parts.items():
            rot_embed = self.head_space_rot_embedders[part](x[..., rot_s:rot_e])
            rot_vel_embed = self.head_space_rot_vel_embedders[part](x[..., rot_vel_s:rot_vel_e])
            pos_embed = self.head_space_pos_embedders[part](x[..., pos_s:pos_e])
            pos_vel_embed = self.head_space_pos_vel_embedders[part](x[..., pos_vel_s:pos_vel_e])

            embeddings.extend([rot_embed, rot_vel_embed, pos_embed, pos_vel_embed])

        # Combine all embeddings
        features = torch.cat(embeddings, dim=-1)

        # Project to ensure exact hidden_size dimension
        features = self.projection(features)

        # Normalize the features
        features = self.norm(features)

        # Remove time dimension if input was a single frame
        if squeeze_output:
            features = features.squeeze(1)

        return features
