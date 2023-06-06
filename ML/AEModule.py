import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dropout

chunksize = 1440 # 1440mins = 1 day

overall_df = pd.read_csv('output/data27_overall.csv')
overall_df = pd.DataFrame(overall_df.iloc[:, 2])
data = overall_df['overall'].values

num_chunks = len(data) // chunksize
data_reshaped = data[:num_chunks * chunksize].reshape(num_chunks, chunksize)
columns = [f'minute_{i+1}' for i in range(chunksize)]

df_reshaped = pd.DataFrame(data_reshaped, columns=columns)

consumption = df_reshaped.values.tolist()
consumption = consumption / np.max(consumption)
train_data, val_data = train_test_split(consumption, test_size=0.2, random_state=42)

layer_dims = [chunksize, 512, 256, 64, 16, 64, 256, 512, chunksize]

input_layer = Input(shape=(layer_dims[0],))
encoder = input_layer
for i in range(1, len(layer_dims) // 2):
    encoder = Dense(layer_dims[i], activation='relu')(encoder)
    encoder = Dropout(0.02)(encoder)

decoder = encoder
for i in range(len(layer_dims) // 2, len(layer_dims)):
    decoder = Dense(layer_dims[i], activation='relu')(decoder)
    decoder = Dropout(0.02)(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, verbose=1)
history = autoencoder.fit(train_data, train_data,
                          validation_data=(val_data, val_data),
                          epochs=300,
                          batch_size=16,
                          callbacks=[early_stopping])

reconstructed_test_data = autoencoder.predict(val_data)
mse = np.mean(np.square(val_data - reconstructed_test_data), axis=1)
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
    plt.savefig(f'/home/mp1820/Y3SummerProject/ML/plots/Best Fits/AE_best_plot_{i+1}.png', dpi=300)
    plt.close()
print("5 Best Reconstructions:")
for i, idx in enumerate(best_indices):
    print(f"Sample {i+1}: MSE = {mse[idx]}")

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
    plt.savefig(f'/home/mp1820/Y3SummerProject/ML/plots/Worst Fits/AE_worst_plot_{i+1}.png', dpi=300)
    plt.close()
print("\n5 Worst Reconstructions:")
for i, idx in enumerate(worst_indices):
    print(f"Sample {i+1}: MSE = {mse[idx]}")