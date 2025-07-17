import math

import torch
import torch.nn as nn


class MultiOmicsTransformer(nn.Module):
    """
    Multi-omics transformer model for integrating heterogeneous biological data types.

    This transformer architecture is specifically designed to handle tabular omics data
    (proteomics, transcriptomics, methylation, etc.) by treating each modality as a
    sequence token and learning cross-modal relationships through self-attention.

    Architecture Overview:
    ---------------------
    1. Each omics modality is embedded into a shared hidden dimension space
    2. Optional modality-specific embeddings are added to distinguish data types
    3. Transformer encoder processes the multi-modal sequence
    4. Attention pooling aggregates information across modalities
    5. Feed-forward prediction head generates final output

    Key Features:
    -------------
    - Handles variable input dimensions across modalities
    - Learnable feature importance weighting per modality
    - Multiple pooling strategies (attention, mean, max, concat)
    - Configurable normalization and activation functions
    - Built-in attention weights for interpretability

    Parameters:
    -----------
    input_dims : dict
        Dictionary mapping modality names to their input dimensions.
        Example: {'proteomics': 1000, 'transcriptomics': 20000, 'methylation': 5000}
    output_dim : int
        Dimension of the final output (e.g., number of classes or regression targets)
    hidden_dim : int, default=256
        Hidden dimension size for transformer embeddings and processing
    num_heads : int, default=8
        Number of attention heads in transformer layers
    num_layers : int, default=6
        Number of transformer encoder layers
    dropout : float, default=0.1
        Dropout probability for regularization
    use_batch_norm : bool, default=True
        Whether to use BatchNorm1d in embedding layers
    use_input_norm : bool, default=True
        Whether to apply LayerNorm to raw inputs
    activation_fn : str or nn.Module, default="gelu"
        Activation function ('gelu', 'relu', 'identity' or PyTorch module)
    use_modality_embedding : bool, default=True
        Whether to add learnable modality-specific embeddings
    pooling_type : str, default="attention"
        Pooling strategy ('attention', 'mean', 'max', 'concat')

    Input Format:
    -------------
    x_dict : dict
        Dictionary with modality names as keys and torch.Tensor as values.
        Each tensor should have shape [batch_size, modality_features].

    Returns:
    --------
    output : torch.Tensor
        Model predictions with shape [batch_size, output_dim]
    attention_weights : torch.Tensor
        Attention weights across modalities with shape [batch_size, num_modalities]
        (only meaningful when pooling_type='attention')

    Example Usage:
    --------------
    >>> input_dims = {"proteomics": 1000, "transcriptomics": 20000, "methylation": 5000}
    >>> model = MultiOmicsTransformer(
    ...     input_dims=input_dims,
    ...     output_dim=2,  # binary classification
    ...     hidden_dim=128,
    ...     num_heads=4,
    ...     num_layers=3,
    ... )
    >>> # Sample input data
    >>> batch_data = {
    ...     "proteomics": torch.randn(32, 1000),
    ...     "transcriptomics": torch.randn(32, 20000),
    ...     "methylation": torch.randn(32, 5000),
    ... }
    >>> predictions, attention_weights = model(batch_data)
    >>> print(f"Predictions shape: {predictions.shape}")  # [32, 2]
    >>> print(f"Attention weights shape: {attention_weights.shape}")  # [32, 3]

    Notes:
    ------
    - The model automatically handles different input dimensions across modalities
    - Feature importance weights are learned during training and applied per modality
    - Attention weights can be used for interpreting which modalities contribute
      most to each prediction
    - Input normalization is recommended for stable training across diverse omics scales
    - The model supports both classification and regression tasks based on output_dim
    """

    def __init__(
        self,
        input_dims,
        output_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        use_batch_norm=True,
        use_input_norm=True,
        activation_fn="gelu",
        use_modality_embedding=True,
        pooling_type="attention",
    ):
        super(MultiOmicsTransformer, self).__init__()

        self.modalities = list(input_dims.keys())
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        self.use_input_norm = use_input_norm
        self.use_modality_embedding = use_modality_embedding
        self.pooling_type = pooling_type
        self.activation_fn = self._get_activation(activation_fn)

        # Optional input normalization (per modality)
        self.input_norms = nn.ModuleDict()
        for omics, input_dim in input_dims.items():
            self.input_norms[omics] = nn.LayerNorm(input_dim) if use_input_norm else nn.Identity()

        # Embedding layers with optional normalization
        self.embeddings = nn.ModuleDict()
        for omics, input_dim in input_dims.items():
            layers = [nn.Linear(input_dim, hidden_dim)]
            layers.append(self._norm_layer(hidden_dim))
            layers.extend([self.activation_fn, nn.Dropout(dropout * 0.5)])
            self.embeddings[omics] = nn.Sequential(*layers)

        self.modality_embeddings = (
            nn.Parameter(torch.randn(len(input_dims), hidden_dim) * 0.02) if use_modality_embedding else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation=activation_fn,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self._norm_layer(hidden_dim * 2),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            self._norm_layer(hidden_dim),
            self.activation_fn,
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, output_dim),
        )

        self.feature_importance = nn.Parameter(torch.ones(len(input_dims)))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _norm_layer(self, dim):
        if self.use_batch_norm:
            return nn.BatchNorm1d(dim)
        elif self.use_input_norm:
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def _get_activation(self, name):
        if isinstance(name, str):
            name = name.lower()
            if name == "gelu":
                return nn.GELU()
            elif name == "relu":
                return nn.ReLU()
            elif name == "identity":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown activation: {name}")
        return name

    def forward(self, x_dict):
        batch_size = next(iter(x_dict.values())).size(0)

        modality_embeddings = []
        feature_weights = torch.softmax(self.feature_importance, dim=0)

        for i, omics in enumerate(self.modalities):
            x = self.input_norms[omics](x_dict[omics])
            x_embedded = self.embeddings[omics](x)

            if self.use_modality_embedding:
                modality_embed = self.modality_embeddings[i].unsqueeze(0).expand(batch_size, -1)
                x_embedded = x_embedded + modality_embed

            x_embedded = x_embedded * feature_weights[i]
            modality_embeddings.append(x_embedded.unsqueeze(1))

        x = torch.cat(modality_embeddings, dim=1)
        x_transformed = self.transformer(x)

        if self.pooling_type == "attention":
            query = self.pooling_query.expand(batch_size, -1, -1)
            pooled, attention_weights = self.attention_pooling(query, x_transformed, x_transformed)
            pooled = pooled.squeeze(1)
        elif self.pooling_type == "mean":
            pooled = x_transformed.mean(dim=1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        elif self.pooling_type == "max":
            pooled, _ = x_transformed.max(dim=1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        elif self.pooling_type == "concat":
            pooled = x_transformed.view(batch_size, -1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        output = self.prediction_head(pooled)

        return output, attention_weights.squeeze(1)


class MultiOmicsTransformerFusion(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        dropout_rate=0.1,
        fusion_method="hierarchical",
        activation_function="gelu",
    ):
        """
        Modular Transformer-based model with various fusion strategies and configurable activation

        Args:
            input_dims: Dictionary mapping omics type to its feature dimension
            output_dim: Number of proteomics features to predict
            fusion_method: One of ["hierarchical", "late", "gated", "weighted", "cross_attention"]
            activation_function: One of ["gelu", "relu"]. PyTorch only allows relu or gelu.
        """
        super(MultiOmicsTransformerFusion, self).__init__()

        self.fusion_method = fusion_method
        self.num_modalities = len(input_dims)

        # Get the specified activation function
        self.activation = self._get_activation_function(activation_function)

        # Create an embedding layer for each omics type
        self.embeddings = nn.ModuleDict(
            {
                omics_type: nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), self.activation)
                for omics_type, dim in input_dims.items()
            }
        )

        # Create separate transformer encoders for each modality
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation=activation_function,
            batch_first=True,
            norm_first=True,
        )

        self.modality_transformers = nn.ModuleDict(
            {omics_type: nn.TransformerEncoder(encoder_layer, num_layers=2) for omics_type in input_dims.keys()}
        )

        # Generate positional encodings once
        self.register_buffer("pos_encoding", self._generate_positional_encoding(len(input_dims), hidden_dim))

        # Type embeddings for all fusion methods
        self.type_embeddings = nn.Embedding(len(input_dims), hidden_dim)

        # Fusion-specific modules
        if fusion_method == "hierarchical":
            # For hierarchical fusion: modality -> joint processing
            self.joint_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif fusion_method == "late":
            # For late fusion: process each modality separately, then combine
            self.fusion_layer = nn.Linear(hidden_dim * len(input_dims), hidden_dim)

        elif fusion_method == "gated":
            # For gated fusion: learn importance of each modality
            self.gate_networks = nn.ModuleDict(
                {
                    omics_type: nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        self.activation,
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Sigmoid(),
                    )
                    for omics_type in input_dims.keys()
                }
            )

        elif fusion_method == "weighted":
            # For weighted fusion: learn a weight for each modality
            self.modality_weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
            self.softmax = nn.Softmax(dim=0)

        elif fusion_method == "cross_attention":
            # For cross-attention: joint transformer with attention pooling
            self.joint_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.attention_pooling = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

        # Final prediction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # expand dimensionality to capture more complex interactions in data
            nn.LayerNorm(hidden_dim * 2),  # normalise activations of preveious layer
            self.activation,  # activation function (ReLU or GeLU)
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),  # reduce dimensionality back, i.e. focus on most important patterns
            nn.LayerNorm(hidden_dim),  # normalise
            self.activation,  # activation function (ReLU or GeLU)
            nn.Dropout(dropout_rate / 2),  # smaller dropout rate to retain more info for the final prediction
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _get_activation_function(self, name):
        """Return the activation function based on name"""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }
        return activations.get(name.lower(), nn.GELU())

    def _generate_positional_encoding(self, seq_len, d_model):
        """Generate positional encodings for the Transformer"""
        pos_encoding = torch.zeros(1, seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def _init_weights(self):
        """Initialise weights using Xavier/Glorot initialisation"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_dict):
        """
        Forward pass

        Args:
            x_dict: Dictionary mapping omics type to tensor of shape [batch_size, features]
        """
        # Determine batch size of input data. As this will be used for various combinations of input data.
        batch_size = next(iter(x_dict.values())).size(0)

        # Process each modality separately first
        modality_features = {}
        embedded_features = []
        modality_indices = []
        for i, (omics_type, x) in enumerate(x_dict.items()):
            # Embed to common dimension
            embedded = self.embeddings[omics_type](x)  # [batch_size, hidden_dim]
            embedded = embedded.unsqueeze(1)  # [batch_size, 1, hidden_dim]

            # Store for fusion methods that need individual modality features
            modality_features[omics_type] = embedded.squeeze(1)  # [batch_size, hidden_dim]

            # Store for fusion methods that need concatenated features
            embedded_features.append(embedded)
            modality_indices.append(i)

            # Process with modality-specific transformer if not using cross_attention
            if self.fusion_method != "cross_attention":
                transformed = self.modality_transformers[omics_type](embedded)
                # Squeeze to remove sequence dimension (only 1 token per modality)
                modality_features[omics_type] = transformed.squeeze(1)  # [batch_size, hidden_dim]

        # Apply fusion method
        if self.fusion_method == "hierarchical" or self.fusion_method == "cross_attention":
            # Concatenate along sequence dimension
            x = torch.cat(embedded_features, dim=1)  # [batch_size, num_omics, hidden_dim]

            # Add positional encoding and type embeddings
            type_ids = torch.tensor(modality_indices, device=x.device).expand(batch_size, -1)
            type_embeds = self.type_embeddings(type_ids)
            x = x + self.pos_encoding.to(x.device) + type_embeds

            # Apply joint transformer
            x = self.joint_transformer(x)  # [batch_size, num_omics, hidden_dim]

            if self.fusion_method == "hierarchical":
                # Take average of all modality representations
                fused = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
            else:  # cross_attention
                # Use attention pooling
                attn_weights = self.attention_pooling(x)  # [batch_size, num_omics, 1]
                fused = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]

        elif self.fusion_method == "late":
            # Concatenate all modality features
            concatenated = torch.cat(
                [modality_features[omics_type] for omics_type in x_dict.keys()], dim=1
            )  # [batch_size, hidden_dim * num_omics]

            # Project back to hidden_dim
            fused = self.fusion_layer(concatenated)  # [batch_size, hidden_dim]

        elif self.fusion_method == "gated":
            # Apply gates to each modality
            gated_features = []
            for omics_type in x_dict.keys():
                gate = self.gate_networks[omics_type](modality_features[omics_type])
                gated = modality_features[omics_type] * gate
                gated_features.append(gated)

            # Sum all gated features
            fused = sum(gated_features)  # [batch_size, hidden_dim]

        elif self.fusion_method == "weighted":
            # Apply learned weights to each modality
            weights = self.softmax(self.modality_weights)

            weighted_sum = None
            for i, omics_type in enumerate(x_dict.keys()):
                if weighted_sum is None:
                    weighted_sum = weights[i] * modality_features[omics_type]
                else:
                    weighted_sum += weights[i] * modality_features[omics_type]

            fused = weighted_sum  # [batch_size, hidden_dim]

        # Final prediction
        features = self.fc_layers(fused)
        output = self.output_layer(features)

        return output
