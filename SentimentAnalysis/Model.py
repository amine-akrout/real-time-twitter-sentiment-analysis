import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
tfds.disable_progress_bar()


data_dir = 'data/arabic_tweets'

batch_size = 32
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    data_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)


raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory( data_dir, batch_size=batch_size)

print("Number of batches in raw_train_ds: %d" % tf.data.experimental.cardinality(raw_train_ds))
print("Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds))
print("Number of batches in raw_test_ds: %d" % tf.data.experimental.cardinality(raw_test_ds))


#####Prepare the data
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

max_features = 5000
embedding_dim = 128
sequence_length = 500


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# Build a model
# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# add embedding layer
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Dense Layer:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# Train the model

epochs = 2
# Fit the model using the train and validation datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
print("Validation Accuracy:  {:.4f}".format(accuracy))
#save model
model.save('sentiment_model.h5')

from tensorflow.keras.models import load_model
m = load_model('sentiment_model.h5')

def sentimentprediction(text):
    clean = custom_standardization(text)
    vec = vectorize_layer(tf.expand_dims(clean, -1))
    if m.predict(vec) > 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment