import keras

import text_format

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

# load data
iterator = text_format.FileIterator(doc_name)
model = iterator.get_model(min_count=min_count, size=embed_size)
dataset = text_format.DataSet(doc_name, embed_size, model=model, min_count=10,
    part_training=0.6, part_validation=0.2)

# get hyperparameters from data
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
autoencoder.compile(loss="categorical_crossentropy", optimizer="rmsprop")

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

# test forward prop
print_samples()

# train
steps_per_epoch = len(dataset) / batch_size
autoencoder.fit_generator(dataset.yield_training_data(batch_size),
    steps_per_epoch, epochs)

# test
print_samples()

# save
autoencoder.save('autoencoder.h5')
