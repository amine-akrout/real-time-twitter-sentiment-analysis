#Import libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import json
# import model and tokenizer
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
clf = load_model('model.h5')

# make a prediction with loaded model
def sentimentprediction(comment):
    data = [comment]
    vect = tokenizer.texts_to_sequences(data)
    vect = pad_sequences(vect, padding='post', maxlen=500)
    prediction = clf.predict_classes(vect)[0][0]
    if prediction > 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment


#comment= "بكون مش عارف دا حلم ولا حقيقه وبعد لما بصحي بدور علي الحاجه اللي حلمت بيها"
