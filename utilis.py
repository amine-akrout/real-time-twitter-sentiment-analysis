import tensorflow as tf 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import load_model
import re
import string
print(tf.__version__)
model = load_model('SentimentAnalysis/sentiment_model.h5')



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

def sentimentprediction(text):
    clean = custom_standardization(text)
    vec = vectorize_layer(tf.expand_dims(clean, -1))
    if model.predict(vec) < 0.5:
        sentiment = "positive"
    else:
        sentiment  = "negative"
    return sentiment
