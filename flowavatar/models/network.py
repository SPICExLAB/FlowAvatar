import torch
import torch.nn as nn

from .blocks import FeatureExtractor, RecurrentBlock


class FlowAvatarNetwork(nn.Module):
    """FlowAvatar pose network: three parallel recurrent branches (upper body,
    lower body, root) over shared embedded features, decoded into SMPL
    parameters (22 joint rotations in 6D + shape betas).

    Optional ``model_params`` keys:
      * ``num_betas`` (default 10): size of the shape head. The released
        streaming checkpoints use 10 (SMPL-X); the AMASS benchmark models
        use 16 (SMPL+H, as in prior work).
      * ``unified_predictor`` (default False): decode all 22 rotations with a
        single MLP instead of the root / upper / lower heads. The benchmark
        models in the paper use the unified decoder; the released streaming
        checkpoints use the per-region heads.

    NOTE on naming: several submodule attributes keep their historical names
    (``*_pose_mamba``, and ``lstm`` inside RecurrentBlock) because they define
    the state_dict keys of the released checkpoints. Do not rename them.
    """
    def __init__(self, configs):
        super().__init__()
        # Extract configuration parameters (``mimik_model_params`` accepted
        # for backward compatibility with older training configs)
        self.model_params = getattr(configs, 'model_params', None) \
            or getattr(configs, 'mimik_model_params', {})

        # Input dimensions
        self.hidden_size = self.model_params.get('hidden_dim', 256)

        # Feature selection configuration
        self.use_head = self.model_params.get('use_head', True)
        self.use_left_hand = self.model_params.get('use_left_hand', True)
        self.use_right_hand = self.model_params.get('use_right_hand', True)
        self.use_left_hand_head_space = self.model_params.get('use_left_hand_head_space', True)
        self.use_right_hand_head_space = self.model_params.get('use_right_hand_head_space', True)
        self.use_head_normalized = self.model_params.get('use_head_normalized', False)

        # Architecture configuration (``mamba_layers`` accepted as legacy key)
        self.num_layers = self.model_params.get(
            'num_layers', self.model_params.get('mamba_layers', 3))
        self.use_lstm = self.model_params.get('use_lstm', False)
        self.num_betas = self.model_params.get('num_betas', 10)
        self.unified_predictor_enabled = self.model_params.get('unified_predictor', False)

        # Set activation class based on model type
        self.activation_class = nn.ReLU if self.use_lstm else nn.SiLU

        # Feature extractor with appropriate activation function
        self.feature_extractor = FeatureExtractor(self.hidden_size, use_lstm=self.use_lstm)

        # Body part specific temporal processors
        if self.num_layers > 1:
            # Upper body blocks
            upper_blocks = [RecurrentBlock(self.hidden_size//2, use_lstm=self.use_lstm) for _ in range(self.num_layers)]
            self.upper_pose_mamba = nn.Sequential(*upper_blocks)

            # Lower body blocks
            lower_blocks = [RecurrentBlock(self.hidden_size//2, use_lstm=self.use_lstm) for _ in range(self.num_layers)]
            self.lower_pose_mamba = nn.Sequential(*lower_blocks)

            # Root blocks
            root_blocks = [RecurrentBlock(self.hidden_size//4, use_lstm=self.use_lstm) for _ in range(self.num_layers)]
            self.root_pose_mamba = nn.Sequential(*root_blocks)
        else:
            # Single layer for each body part
            self.upper_pose_mamba = RecurrentBlock(self.hidden_size//2, use_lstm=self.use_lstm)
            self.lower_pose_mamba = RecurrentBlock(self.hidden_size//2, use_lstm=self.use_lstm)
            self.root_pose_mamba = RecurrentBlock(self.hidden_size//4, use_lstm=self.use_lstm)

        # Body part specific processors
        self.upper_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            self.activation_class(),
            nn.LayerNorm(self.hidden_size//2)
        )

        self.lower_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            self.activation_class(),
            nn.LayerNorm(self.hidden_size//2)
        )

        self.root_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//4),
            self.activation_class(),
            nn.LayerNorm(self.hidden_size//4)
        )

        # Integration module
        combined_dim = self.hidden_size//2 + self.hidden_size//2 + self.hidden_size//4
        self.integration = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            self.activation_class(),
            nn.LayerNorm(self.hidden_size)
        )

        if self.unified_predictor_enabled:
            # Single decoder for all 22 joints (benchmark models)
            self.unified_predictor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                self.activation_class(),
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, 22*6)
            )
        else:
            # Rotation predictors for different body parts
            self.root_predictor = nn.Sequential(
                nn.Linear(self.hidden_size//3, 64),
                self.activation_class(),
                nn.LayerNorm(64),
                nn.Linear(64, 6)  # 6D rotation for root
            )

            # Upper body rotation predictor for 13 joints
            self.upper_body_predictor = nn.Sequential(
                nn.Linear(self.hidden_size//3, self.hidden_size//3),
                self.activation_class(),
                nn.LayerNorm(self.hidden_size//3),
                nn.Linear(self.hidden_size//3, 13*6)  # 6D rotation for 13 upper body joints
            )

            # Lower body rotation predictor for 8 joints
            self.lower_body_predictor = nn.Sequential(
                nn.Linear(self.hidden_size//3, self.hidden_size//3),
                self.activation_class(),
                nn.LayerNorm(self.hidden_size//3),
                nn.Linear(self.hidden_size//3, 8*6)  # 6D rotation for 8 lower body joints
            )

        # Shape predictor
        # Use LeakyReLU for shape prediction regardless of model type for better stability
        self.shape_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_betas)
        )

    def apply_feature_masking(self, x_in):
        """
        Apply feature masking based on configuration
        """
        # Create a copy to avoid modifying the original
        x = x_in.clone()

        # Apply spatial normalization if enabled
        if self.use_head_normalized:
            # Store original values
            x_original = x.clone()

            # Normalize head position by zeroing out x,y coordinates (indices 36, 37)
            x[..., 36:38] = x[..., 36:38] - x_original[..., 36:38]

        # Apply masking based on configuration - using correct indices for 90-dim input
        if not self.use_head:
            # Zero out head features
            x[..., 0:6] = 0      # Rotation
            x[..., 18:24] = 0    # Rotation velocity
            x[..., 36:39] = 0    # Position
            x[..., 45:48] = 0    # Position velocity

        if not self.use_left_hand:
            # Zero out left hand features
            x[..., 6:12] = 0     # Rotation
            x[..., 24:30] = 0    # Rotation velocity
            x[..., 39:42] = 0    # Position
            x[..., 48:51] = 0    # Position velocity

        if not self.use_right_hand:
            # Zero out right hand features
            x[..., 12:18] = 0    # Rotation
            x[..., 30:36] = 0    # Rotation velocity
            x[..., 42:45] = 0    # Position
            x[..., 51:54] = 0    # Position velocity

        if not self.use_left_hand_head_space:
            x[..., 54:60] = 0    # Rotation
            x[..., 66:72] = 0    # Rotation velocity
            x[..., 78:81] = 0    # Position
            x[..., 84:87] = 0    # Position velocity

        if not self.use_right_hand_head_space:
            x[..., 60:66] = 0    # Rotation
            x[..., 72:78] = 0    # Rotation velocity
            x[..., 81:84] = 0    # Position
            x[..., 87:90] = 0    # Position velocity

        return x

    def forward(self, x_in):
        """
        Forward pass through the network

        Args:
            x_in: Input features [B, T, 90]

        Returns:
            Dictionary with pose_params [B, T, 132] and betas [B, T, num_betas]
        """
        batch_size, time_seq = x_in.shape[0], x_in.shape[1]

        # Apply feature masking
        x_masked = self.apply_feature_masking(x_in)

        # Extract features
        features = self.feature_extractor(x_masked)

        # Process body parts
        upper_features = self.upper_processor(features)
        lower_features = self.lower_processor(features)
        root_features = self.root_processor(features)

        # Temporal processing per body region
        upper_processed = self.upper_pose_mamba(upper_features)
        lower_processed = self.lower_pose_mamba(lower_features)
        root_processed = self.root_pose_mamba(root_features)

        # Combine processed features
        combined_processed = torch.cat([upper_processed, lower_processed, root_processed], dim=-1)
        integrated = self.integration(combined_processed)

        if self.unified_predictor_enabled:
            all_rotations = self.unified_predictor(integrated)
        else:
            # Split integrated features for different body parts
            integrated_split = torch.split(integrated, self.hidden_size//3, dim=-1)

            # Predict rotations
            root_rotation = self.root_predictor(integrated_split[0])
            upper_body_rotations = self.upper_body_predictor(integrated_split[1])
            lower_body_rotations = self.lower_body_predictor(integrated_split[2])

            # Combine all rotations
            all_rotations = torch.cat([root_rotation, upper_body_rotations, lower_body_rotations], dim=-1)

        # Calculate shape parameters (average across time for stability)
        time_pooled = torch.mean(integrated, dim=1)
        shapes = self.shape_predictor(time_pooled)

        # Replicate shapes across time dimension
        pred_shapes = shapes.unsqueeze(1).expand(-1, time_seq, -1)

        # Return as dictionary
        return {
            "pose_params": all_rotations,
            "betas": pred_shapes
        }

    def forward_offline(self, x):
        """
        Inference entry point used by the streaming demo: returns only the
        last frame's prediction, split into the parts Unity expects.

        Args:
            x: Input features [B, T, 90]

        Returns:
            Dictionary with root_rot [B, 6], body_rot [B, 126], betas [B, num_betas]
        """
        with torch.inference_mode():
            # Get outputs from forward pass
            outputs = self(x)

            # Extract pose parameters for the last frame
            pose_params = outputs["pose_params"][:, -1]  # [B, 132]

            # Split into root and body rotations
            root_rot = pose_params[:, :6]  # First 6 values are root rotation
            body_rot = pose_params[:, 6:]  # Remaining values are body rotation

            return {
                'body_rot': body_rot,
                'root_rot': root_rot,
                "betas": outputs["betas"][:, -1]
            }
