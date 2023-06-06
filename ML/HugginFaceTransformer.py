import pandas as pd
import torch
import numpy as np
from torch import Tensor
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction, AdamW

# Load your data
data = pd.read_csv('output/data27_overall.csv')

# Parse the 'localminute' column to datetime format
data['localminute'] = pd.to_datetime(data['localminute'])

# Define your model configuration
config = TimeSeriesTransformerConfig(
    prediction_length=10,
    context_length=60,
    distribution_output='student_t',
    loss='nll',
    input_size=1,
    scaling='mean',
    lags_sequence=[1, 2, 3, 4, 5, 6, 7],
    num_time_features=2,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=32,
    decoder_ffn_dim=32,
    dropout=0.1
)

def create_sequences(input_data, tw, is_feature=False):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1] if not is_feature else None
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Extract features and values
values = data['overall'].values
features = data['localminute'].astype(int).values.reshape(-1, 1)

# The window size should be greater than the context_length defined in the config
window_size = config.context_length + config.prediction_length

# Create sequences
past_values_seq = create_sequences(values, window_size)
future_values_seq = create_sequences(values, config.prediction_length)

# Repeat time features to have the same number of channels as values
features = np.repeat(features, config.input_size, axis=-1)
past_time_features_seq = create_sequences(features, window_size, is_feature=True)
future_time_features_seq = create_sequences(features, config.prediction_length, is_feature=True)

# Create observed mask
past_observed_mask = torch.ones(len(past_values_seq), window_size)
future_observed_mask = torch.ones(len(future_values_seq), config.prediction_length)

# Ensure all tensors have the right dimensions
past_values = torch.FloatTensor([pair[0] for pair in past_values_seq]).unsqueeze(-1)
past_time_features = torch.FloatTensor([pair[0] for pair in past_time_features_seq])
future_values = torch.FloatTensor([pair[0] for pair in future_values_seq]).unsqueeze(-1)
future_time_features = torch.FloatTensor([pair[0] for pair in future_time_features_seq])

# Create the model
model = TimeSeriesTransformerForPrediction(config)

# Create the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Model training
for i in range(len(past_values)):
    model.train()
    optimizer.zero_grad()

    outputs = model(
        past_values=past_values[i],
        past_time_features=past_time_features[i],
        past_observed_mask=past_observed_mask[i],
        future_values=future_values[i],
        future_time_features=future_time_features[i],
        future_observed_mask=future_observed_mask[i]
    )

    # Compute the loss
    loss = outputs.loss
    print(f"Loss at step {i}: {loss}")

    # Backpropagate the loss
    loss.backward()

    # Update weights
    optimizer.step()