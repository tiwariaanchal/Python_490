import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn.preprocessing import LabelEncoder

from keras.models import load_model
model = load_model('model.h5')

new_text = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"

new_string = [[new_text]]
max_df = pd.DataFrame(new_string, index=range(0,1,1), columns=list('t'))

max_df['t'] = max_df['t'].apply(lambda x: x.lower())
max_df['t'] = max_df['t'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
print(max_df)


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(max_df['t'].values)
X = tokenizer.texts_to_sequences(max_df['t'].values)

X = pad_sequences(X, maxlen=28)
print(model.predict(X))

