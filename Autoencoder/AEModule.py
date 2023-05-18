import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping

overall_df = pd.read_csv('output/data27_overall.csv')
overall_df = pd.DataFrame(overall_df.iloc[:, 2])

# Assuming your DataFrame is named 'df'
data = overall_df['overall'].values

# Reshape the data into chunks of 1440 measurements
num_chunks = len(data) // 1440
data_reshaped = data[:num_chunks * 1440].reshape(num_chunks, 1440)

# Convert the reshaped data back into a DataFrame
columns = [f'minute_{i+1}' for i in range(1440)]
df_reshaped = pd.DataFrame(data_reshaped, columns=columns)

consumption = df_reshaped.values.tolist()

# Normalize the data to be in the range [0, 1]
consumption = consumption / np.max(consumption)

# Split the data into training and validation sets
train_data, val_data = train_test_split(consumption, test_size=0.2, random_state=42)

# Define the dimensions of the layers in the autoencoder
layer_dims = [1440, 512, 256, 64, 16, 64, 256, 512, 1440]

# Define the input layer
input_layer = Input(shape=(layer_dims[0],))

# Build the encoder layers
encoder = input_layer
for i in range(1, len(layer_dims) // 2):
    encoder = Dense(layer_dims[i], activation='relu')(encoder)

# Build the decoder layers
decoder = encoder
for i in range(len(layer_dims) // 2, len(layer_dims)):
    decoder = Dense(layer_dims[i], activation='relu')(decoder)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1)

# Train the autoencoder with early stopping
history = autoencoder.fit(train_data, train_data,
                          validation_data=(val_data, val_data),
                          epochs=80,
                          batch_size=16,
                          callbacks=[early_stopping])


# Reconstruct the test data using the trained autoencoder
reconstructed_test_data = autoencoder.predict(val_data)

# Calculate the mean squared error for each sample in the test data
mse = np.mean(np.square(val_data - reconstructed_test_data), axis=1)

# Find the indices of the 5 best and 5 worst reconstructions
best_indices = np.argsort(mse)[:5]
worst_indices = np.argsort(mse)[-5:]

sns.set(style='whitegrid', font_scale=1.2)

# Plot the 5 best reconstructions
for i, idx in enumerate(best_indices):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(val_data[idx])), val_data[idx], label='Truth', linestyle='-', linewidth=2)
    plt.plot(range(len(reconstructed_test_data[idx])), reconstructed_test_data[idx], label='Reconstructed', linestyle='--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Time Series Plot (Best) {i+1}\nMSE: {mse[idx]}\nIndex: {idx}', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylim(0, 0.7)  # Set y-axis limits
    plt.xticks(np.arange(0, len(val_data[idx]), 60), np.arange(0, len(val_data[idx])//60))
    plt.tight_layout()
    plt.savefig(f'AE_best_plot_{i+1}.png', dpi=300)
    plt.close()

# Plot the 5 worst reconstructions
for i, idx in enumerate(worst_indices):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(val_data[idx])), val_data[idx], label='Truth', linestyle='-', linewidth=2)
    plt.plot(range(len(reconstructed_test_data[idx])), reconstructed_test_data[idx], label='Reconstructed', linestyle='--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Time Series Plot (Worst) {i+1}\nMSE: {mse[idx]}\nIndex: {idx}', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylim(0, 0.7)  # Set y-axis limits
    plt.xticks(np.arange(0, len(val_data[idx]), 60), np.arange(0, len(val_data[idx])//60))
    plt.tight_layout()
    plt.savefig(f'AE_worst_plot_{i+1}.png', dpi=300)
    plt.close()


# Print the MSE for the 5 best and 5 worst reconstructions
print("5 Best Reconstructions:")
for i, idx in enumerate(best_indices):
    print(f"Sample {i+1}: MSE = {mse[idx]}")

print("\n5 Worst Reconstructions:")
for i, idx in enumerate(worst_indices):
    print(f"Sample {i+1}: MSE = {mse[idx]}")


