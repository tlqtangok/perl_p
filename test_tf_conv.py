### a really simple example to use libcudnn and plot loss via id_pd.plot.scatter

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, Conv2D, Flatten, Reshape
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

from functools import reduce
import tarfile
import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image
#import matplotlib.pyplot as plt

from tensorflow.keras import backend as K




id_shape = [10000, 28,28,1]
x_train_all =  np.random.random(id_shape)
y_label_all =  x_train_all * 3.3 + 4.4

x_train, y_test, x_label, y_label = train_test_split(x_train_all, y_label_all, test_size=0.2)

model = Sequential(
[
    Conv2D(4, (3,3), padding="same", activation="relu", input_shape=x_train.shape[1:]), 
    Dense(1)
]
)

model.summary()

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    x_train, x_label, 
    validation_data=(y_test, y_label),
    epochs=32,
    batch_size=1024,
    verbose=1
)

id_pd = pd.DataFrame(history.history)


id_pd["idx"] = np.arange(len(id_pd))

# display(id_pd)

#id_pd.plot.scatter(x=["idx"] * 2,y=["loss", "val_loss"], c=["blue", "red"])
#plt.show()

#K.clear_session()

