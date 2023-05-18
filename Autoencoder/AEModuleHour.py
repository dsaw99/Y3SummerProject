import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
import matplotlib.pyplot as plt

csvfile = 'output/overall_consumption.csv'
overall_df = pd.read_csv(csvfile)
overall_df = pd.DataFrame(overall_df.iloc[:, 2]) 

# Assuming your DataFrame is named 'df'
data = overall_df['overall'].values

# Reshape the data into chunks of 60 measurements
num_chunks = len(data) // 60
data_reshaped = data[:num_chunks * 60].reshape(num_chunks, 60)

# Convert the reshaped data back into a DataFrame
columns = [f'minute_{i+1}' for i in range(60)]
df_reshaped = pd.DataFrame(data_reshaped, columns=columns)

print(df_reshaped)

consumption = df_reshaped.values.tolist()

# Normalize the data to be in the range [0, 1]
consumption = consumption / np.max(consumption)

# Split the data into training and validation sets
train_data, val_data = train_test_split(consumption, test_size=0.2, random_state=42)

# Define the dimensions of the layers in the autoencoder
layer_dims = [60, 120, 40, 20, 40, 120, 60]

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

# Train the autoencoder
history = autoencoder.fit(train_data, train_data,
                          validation_data=(val_data, val_data),
                          epochs=150,
                          batch_size=16)




# Reconstruct the data using the trained autoencoder
reconstructed_data = autoencoder.predict(consumption)

time = range(len(reconstructed_data[1]))

# Plot the time series
plt.plot(time, reconstructed_data[1])

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Plot')

# Display the plot
plt.savefig('AE')
plt.close

time = range(len(consumption[1]))

# Plot the time series
plt.plot(time, consumption[1])

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Plot')

# Display the plot
plt.savefig('AE_true')
plt.close




# Calculate the reconstruction loss
mse = np.mean(np.square(consumption - reconstructed_data))
print('Mean Squared Error (MSE):', mse)