import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_original_dataset(root):
    train_path = os.path.join(root, 'train')

    files = os.listdir(os.path.join(train_path, 'pos'))
    pos_review = list()
    for file in files:
        with open(os.path.join(train_path, 'pos', file), encoding="latin1") as f:
            pos_review.append(f.read())
    pos_label = [1] * len(pos_review)

    files = os.listdir(os.path.join(train_path, 'neg'))
    neg_review = list()
    for file in files:
        with open(os.path.join(train_path, 'neg', file), encoding="latin1") as f:
            neg_review.append(f.read())
    neg_label = [0] * len(neg_review)
    data = pd.DataFrame({'Review': pos_review + neg_review, 'Target': pos_label + neg_label})
    return data


def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


root = os.path.join(os.getcwd(), 'database')
# df = load_original_dataset(root=root)
df = pd.read_csv(os.path.join(root, 'imdb_master.csv'),  encoding="latin1")
df.drop(columns=['Unnamed: 0', 'type', 'file'], inplace=True)
df_clean = df.loc[~(df.label == 'unsup'), :].copy()
df_clean.loc[:, 'label'] = df_clean.label.map({'pos': 1, 'neg': 0})
df_clean['ProcessedReviews'] = df_clean.review.apply(lambda x: clean_text(x))

rand_idx = np.random.randint(0, df_clean.shape[0], df_clean.shape[0])
split_val = int(rand_idx.shape[0] * .7)
train_set = df_clean.iloc[rand_idx[:split_val]]
test_set = df_clean.iloc[rand_idx[split_val + 1:]]

seq_length = int(train_set.apply(lambda x: len(x.ProcessedReviews.split(' ')), axis=1).mean() + 10)
# create a tokenizer for the reviews
tokenizer_obj = Tokenizer(lower=True, split=' ')
tokenizer_obj.fit_on_texts(train_set.ProcessedReviews)
# fit on data
tokenizer_obj.num_words = 6000
x_train = tokenizer_obj.texts_to_sequences(train_set.ProcessedReviews)
x_test = tokenizer_obj.texts_to_sequences(test_set.ProcessedReviews)


x_train_norm = pad_sequences(x_train, maxlen=seq_length)
x_test_norm = pad_sequences(x_test, maxlen=seq_length)
y_train = train_set.label
y_test = test_set.label

input_nn = Input(shape=(x_train_norm.shape[1], ))
embedding = Embedding(6000, 256, input_length=seq_length, name='embedding')(input_nn)
flatten = Flatten()(embedding)
hidden = Dense(units=256, activation='relu', name='hidden_1')(flatten)
hidden = Dropout(rate=0.25, name='dropout_1')(hidden)
hidden = Dense(units=128, activation='relu', name='hidden_2')(flatten)
output_nn = Dense(units=1, activation='sigmoid', name='output')(hidden)

model_nn = Model(inputs=input_nn, outputs=output_nn)
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.optimizer.lr = 0.01
model_nn.fit(x=x_train_norm, y=y_train, batch_size=128,  epochs=2, validation_data=(x_test_norm, y_test))

book_layer = model_nn.get_layer('embedding')
book_weights = book_layer.get_weights()[0]
book_weights.shape



pos_review_trans = tokenizer_obj.texts_to_sequences(df.loc[df.Target == 1, 'Review'])
neg_review_trans = tokenizer_obj.texts_to_sequences(df.loc[df.Target == 0, 'Review'])

# get the number of words from the longest review
res = df.apply(lambda x: len(x.Review.split(' ')), axis=1)
max_length = res.max()

# get information about the mapping and the length of it
dictionary = tokenizer_obj.word_index
dictionary_size = len(tokenizer_obj.word_index) + 1
print('Longest Review:', max_length, 'Dict Size:', dictionary_size)

# we need to normalize the length of each review (everyone must be the same length)
# sequences is a list of integers,
# maxlen is the new length for each list,
# padding define where "value" are padded
pos_review_norm = pad_sequences(sequences=pos_review_trans,
                                maxlen=max_length,
                                padding='post',
                                value=0)

neg_review_norm = pad_sequences(sequences=neg_review_trans,
                                maxlen=max_length,
                                padding='post',
                                value=0)

# join dataset
x_train_final = np.concatenate((pos_review_norm, neg_review_norm), axis=0)
y_train_final = df.Target.values
# re-order data
idx = np.random.randint(0, y_train_final.shape[0], y_train_final.shape[0])
x_train_final_1 = x_train_final[idx]
y_train_final_1 = y_train_final[idx]

# create network
input_nn = Input(shape=(x_train_final.shape[1], ))
embedding = Embedding(dictionary_size, 128, input_length=max_length)(input_nn)
flatten = Flatten()(embedding)
hidden = Dense(units=128, activation='relu', name='hidden_1')(flatten)
# hidden = Dense(units=1024, activation='relu', name='hidden_2')(hidden)
# hidden = Dense(units=1024, activation='relu', name='hidden_3')(hidden)
output_nn = Dense(units=1, activation='sigmoid', name='output')(hidden)

model_nn = Model(inputs=input_nn, outputs=output_nn)
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.optimizer.lr = 0.01
model_nn.fit(x=x_train_final, y=y_train_final, epochs=3, validation_data=(x_train_final_1, y_train_final_1))








