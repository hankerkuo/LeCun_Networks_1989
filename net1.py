import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from six.moves import cPickle as pickle
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
np.random.seed(1337)  # for reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scaled_hyperbolic_tangent(x):
    return K.tanh((2 / 3) * x) * 1.7159


get_custom_objects().update({'scaled_hyperbolic_tangent': Activation(scaled_hyperbolic_tangent)})

with open('./train_data/data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open('./train_data/label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open('./test_data/data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open('./test_data/label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

# data pre-processing
tr_dat = tr_dat.reshape(-1, 16 * 16)
te_dat = te_dat.reshape(-1, 16 * 16)

model = Sequential()

# Fully connected layer 1 to shape (10) for 10 classes
model.add(Dense(
        units=10,
        batch_input_shape=(None, 16 * 16),))
model.add(Activation('scaled_hyperbolic_tangent'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='mse',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(tr_dat, tr_lab, epochs=3000, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(te_dat, te_lab)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
