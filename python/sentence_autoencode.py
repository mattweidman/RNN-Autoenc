import keras

import text_format

doc_name = "../shakespeare/shakes_out.txt"

# hyperparameters
min_count = 10
embed_size = 100
enc_lstm_size = 50
autoenc_embed_size = 20

# load data
iterator = text_format.FileIterator(doc_name)
model = iterator.get_model(min_count=min_count, size=embed_size)
dataset = text_format.DataSet(doc_name, embed_size, model=model,
    part_training=0.6, part_validation=0.2)
minibatch = dataset.get_training_data(10)

# get more parameters from data
seq_len = dataset.max_line_len

# build encoder model
enc_input = keras.layers.Input(shape=[seq_len, embed_size])
H = keras.layers.LSTM(enc_lstm_size, return_sequences=False) \
    (enc_input)
H = keras.layers.Dropout(0.2)(H)
enc_output = keras.layers.Dense(autoenc_embed_size, activation="softmax")(H)
encoder = keras.models.Model(enc_input, enc_output)
encoder.compile(loss="categorical_crossentropy", optimizer="adam")

# test encoder model
prediction = encoder.predict(minibatch)
print(prediction)
