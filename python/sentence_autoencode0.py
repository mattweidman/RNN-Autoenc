import keras
import numpy as np

doc_name = "../shakespeare/shakes_out.txt"

# hyperparameters
min_count = 2 # 10 yields vocab of size 5719 on Shakespeare dataset
embed_size = 250
enc_lstm_size = 250
dec_lstm_size = 250
layer_size = 250
encoder_size = 1200
dropout_rate = 0.2
epochs = 8
batch_size = 128
sample_size = 4

# load lines
with open(doc_name, 'r') as rfile:
    lines = rfile.readlines()

# sorted list of words in vocabulary
vocab = []
for line in lines: vocab += line.split()
vocab = sorted(set(vocab))

# word -> vector mapping
class WordToOnehot:
    def __init__(self, vocab):
        self.word_to_index = dict((s,i) for i,s in enumerate(vocab))
    def __getitem__(self, word):
        if word not in self.word_to_index:
            raise KeyError(str(word) + " has no one-hot representation")
        index = self.word_to_index[word]
        vec = np.zeros((len(self.word_to_index)))
        vec[index] = 1
        return vec
word_to_onehot = WordToOnehot(vocab)

# vector -> word mapping
class OnehotToWord:
    def __init__(self, vocab):
        self.vocab = vocab
    def __getitem__(self, vector):
        if vector.shape != (len(vocab),):
            raise KeyError("Vector should be size of vocabulary. Expected "\
                + str((len(vocab),)) + " but got " + str(vector.shape))
        index = np.argmax(vector)
        return self.vocab[index]
onehot_to_word = OnehotToWord(vocab)

for word in vocab:
    assert onehot_to_word[word_to_onehot[word]] == word

'''# get hyperparameters from data
seq_len = dataset.max_line_len
num_words = len(dataset.word_list)

# build encoder model
enc_input = keras.layers.Input(shape=[seq_len, embed_size])
H = keras.layers.LSTM(enc_lstm_size, return_sequences=True,
    dropout=dropout_rate)(enc_input)
H = keras.layers.LSTM(enc_lstm_size, return_sequences=False,
    dropout=dropout_rate)(H)
enc_output = keras.layers.Dense(encoder_size, activation='softmax')(H)

# build decoder model
H = keras.layers.RepeatVector(seq_len)(enc_output)
H = keras.layers.LSTM(dec_lstm_size, return_sequences=True,
    dropout=dropout_rate)(H)
H = keras.layers.LSTM(dec_lstm_size, return_sequences=True,
    dropout=dropout_rate)(H)
dec_output = keras.layers.Dense(num_words, activation='sigmoid')(H)

# finalize autoencoder
autoencoder = keras.models.Model(enc_input, dec_output)
autoencoder.compile(loss="categorical_crossentropy", optimizer="adam")

# save picture of model
keras.utils.plot_model(autoencoder, to_file='../model_vis.png',
    show_shapes=True)

# displays expected vs actual autoencoder output
def print_samples():
    batch_x, batch_y = dataset.get_training_data(sample_size)
    prediction = autoencoder.predict(batch_x)
    print("Expected:")
    print(dataset.long_tensor_to_string(batch_y))
    print(batch_y.shape)
    print("Output:")
    print(dataset.long_tensor_to_string(prediction))
    print(prediction.shape)

# saving/displaying checkpoint
weightspath = "../weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weightspath, monitor="loss",
    verbose=1, save_best_only=False, mode="min")
print_samples_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print_samples())

# train
steps_per_epoch = len(dataset) / batch_size
autoencoder.fit_generator(dataset.yield_training_data(batch_size),
    steps_per_epoch, epochs, callbacks=[checkpoint, print_samples_callback])
'''
