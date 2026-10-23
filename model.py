"""Core model extracted from archived research code; see project.json and CHANGELOG.md."""
import numpy as np
import torch
from torch import nn, Tensor, cat, matmul, squeeze, floor
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cpu")
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, time_steps, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        
        # Project input to d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, time_steps, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        # Multi-layer structure for the output classifier
        self.fc = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_classes)
        )

    def _initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [batch_size, time_steps, input_dim]
        batch_size = x.size(0)
        
        # Project input to d_model dimension
        x = self.input_proj(x.float())  # [batch_size, time_steps, d_model]
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :self.time_steps, :]
        
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, time_steps, d_model] because batch_first=True
        
        # Use both global average pooling and global max pooling over the time dimension, then concatenate
        x_avg = x.mean(dim=1)         # [batch_size, d_model]
        x_max, _ = x.max(dim=1)       # [batch_size, d_model]
        x = torch.cat([x_avg, x_max], dim=1)  # [batch_size, 2*d_model]
        # Also update output layer's input dim accordingly (fc = nn.Linear(2*d_model, num_classes))
        
        output = self.fc(x)
        return output

def add_physics_features(x_data):
    """
    Add physics-informed features to the input data.
    
    Assumes column order: origin_wind_speed, final_wind, peak_current, 
                          rotational_speed, Tambient, TIGBT_junction, Tcase
    
    Parameters:
    -----------
    x_data : numpy array
        Input features of shape (n_samples, 7)
    
    Returns:
    --------
    x_with_features : numpy array
        Features with physics-informed features added, shape (n_samples, 10)
    """
    epsilon = 1e-6  # Small constant to avoid division by zero
    k_calibration = 1.0  # Calibration constant for R_th_estimate
    
    # Extract relevant columns
    # Column indices: 0=origin_wind, 1=final_wind, 2=peak_current, 3=rotational_speed,
    #                 4=Tambient, 5=TIGBT_junction, 6=Tcase
    peak_current = x_data[:, 2]
    T_IGBT_junction = x_data[:, 5]
    T_case = x_data[:, 6]
    
    # 1. Junction-to-Case Temperature Difference
    delta_T_jc = T_IGBT_junction - T_case
    
    # 2. Normalized Temperature Differential
    delta_T_norm = delta_T_jc / (peak_current + epsilon)
    
    # 3. Instantaneous Thermal Resistance Estimate
    # R_th_estimate = (T_junction - T_case) / (I² × k + ε)
    I_squared = peak_current ** 2
    R_th_estimate = delta_T_jc / (I_squared * k_calibration + epsilon)
    
    # Concatenate original features with physics-informed features
    x_with_features = np.concatenate([
        x_data,  # Original 7 features
        delta_T_jc.reshape(-1, 1),      # Feature 8
        delta_T_norm.reshape(-1, 1),    # Feature 9
        R_th_estimate.reshape(-1, 1)    # Feature 10
    ], axis=1)
    
    return x_with_features
