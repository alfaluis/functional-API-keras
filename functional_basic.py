from keras.datasets import cifar10
from keras.models import Input, Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
im_flatten_size = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
x_train_new = x_train.reshape((x_train.shape[0], 3072))
x_test_new = x_test.reshape((x_test.shape[0], 3072))

# z-score for features
std_scaler = StandardScaler().fit(x_train_new)
x_train_new = std_scaler.transform(x_train_new)
x_test_new = std_scaler.transform(x_test_new)

# label as one-hot encoding
lbl_binarize = LabelBinarizer().fit(y_train)
y_train_bin = lbl_binarize.transform(y_train)
y_test_bin = lbl_binarize.transform(y_test)

# Build the model with Functional API
inputs = Input(shape=(3072,))
output = Dense(10, activation='softmax')(inputs)
logistic_model = Model(inputs, output)

# Compile the model
logistic_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Fit on training data
logistic_model.optimizer.lr = 0.001
logistic_model.fit(x=x_train_new, y=y_train_bin, epochs=5, validation_data=(x_test_new, y_test_bin))

# training a neural network
input_nn = Input(shape=(3072, ))
hidden = Dense(units=512, activation='relu', name='hidden_1')(input_nn)
hidden = Dense(units=128, activation='relu', name='hidden_2')(hidden)
output_nn = Dense(units=10, activation='softmax', name='output')(hidden)

model_nn = Model(inputs=input_nn, outputs=output_nn)
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
logistic_model.optimizer.lr = 0.001
logistic_model.fit(x=x_train_new, y=y_train_bin, epochs=5, validation_data=(x_test_new, y_test_bin))


