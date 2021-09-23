import pandas as pd
import numpy as np
import tensorflow as tf

train_pos = pd.read_csv('SentimentAnalysis/data/train_Arabic_tweets_positive_20190413.tsv', sep='\t', header=None)
train_neg = pd.read_csv('SentimentAnalysis/data/train_Arabic_tweets_negative_20190413.tsv', sep='\t', header=None)
test_pos = pd.read_csv('SentimentAnalysis/data/test_Arabic_tweets_positive_20190413.tsv', sep='\t', header=None)
test_neg = pd.read_csv('SentimentAnalysis/data/test_Arabic_tweets_negative_20190413.tsv', sep='\t', header=None)

train = pd.concat([train_pos, train_neg])
test = pd.concat([test_pos, test_neg])

train.columns = ['label', 'text']
test.columns = ['label', 'text']


train['label'] = np.where(train['label']=='neg',0,1)
test['label'] = np.where(test['label']=='neg',0,1)

X_train, y_train, X_test, y_test = train['text'], train['label'], test['text'], test['label']


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = 500
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

embedding_dim = 128

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
model.add(Dropout(0.5))
model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
model.add(layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## Fit the model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size = 64, callbacks=[es])

#summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



###Test model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#save tokenizer
import json
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
#save model
from tensorflow.keras.models import save_model, load_model
save_model(model,'model.h5')  # creates a HDF5 file 'my_model.h5'


# test if it works
clf = load_model('model.h5')
comment = "بكون مش عارف دا حلم ولا حقيقه وبعد لما بصحي بدور علي الحاجه اللي حلمت بيها"

def predict(comment):
    data = [comment]
    vect = tokenizer.texts_to_sequences(data)
    vect = pad_sequences(vect, padding='post', maxlen=500)
    prediction = clf.predict_classes(vect)[0][0]
    if prediction > 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return sentiment


