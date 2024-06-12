
import tensorflow as tf
import tensorflow.hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

data = ["This is the first sentence.", "This is the second sentence.", "Tokenization is important for natural language processing."]

tokenizer = Tokenizer()
sequence_data = tokenizer.texts_to_sequences(data)

tfmodel = Sequential()
tfmodel.add(layers.Embedding(input_dim=1000, output_dim=64))
tfmodel.add(layers.GlobalAveragePooling1D())
tfmodel.add(layers.Dense(1, activation='sigmoid'))

tfmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tfmodel.fit(sequence_data, np.array([1, 0, 1]), epochs=10)
