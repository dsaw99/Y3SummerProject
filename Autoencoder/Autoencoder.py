from keras.models import Model
from keras.layers import Input, Dense


input_dim = 1440
bottleneck_dim = 2

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(500, activation='relu')(input_layer)
encoder = Dense(200, activation='relu')(encoder)
encoder = Dense(100, activation='relu')(encoder)
encoder = Dense(bottleneck_dim, activation='relu')(encoder)

# Decoder
decoder = Dense(100, activation='relu')(encoder)
decoder = Dense(200, activation='relu')(decoder)
decoder = Dense(500, activation='relu')(decoder)
decoder = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.summary()