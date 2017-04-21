import keras

import text_format

doc_name = "../shakespeare/shakes_out.txt"

# hyperparameters
min_count = 10
embed_size = 100
enc_lstm_size = 50
dec_lstm_size = 50
autoenc_embed_size = 20
dropout_rate = 0.2

# load data
iterator = text_format.FileIterator(doc_name)
model = iterator.get_model(min_count=min_count, size=embed_size)
dataset = text_format.DataSet(doc_name, embed_size, model=model,
    part_training=0.6, part_validation=0.2)
minibatch = dataset.get_training_data(10)

# get hyperparameters from data
seq_len = dataset.max_line_len

# build encoder model
enc_input = keras.layers.Input(shape=[seq_len, embed_size])
H = keras.layers.LSTM(enc_lstm_size, return_sequences=False) \
    (enc_input)
H = keras.layers.Dropout(dropout_rate)(H)
enc_output = keras.layers.Dense(autoenc_embed_size, activation="softmax")(H)
encoder = keras.models.Model(enc_input, enc_output)
encoder.compile(loss="categorical_crossentropy", optimizer="adam")

# test encoder model
autoenc_vectors = encoder.predict(minibatch)
print(autoenc_vectors.shape)

# build decoder model
dec_input = keras.layers.Input(shape=[autoenc_embed_size])
H = keras.layers.Dense(dec_lstm_size, activation="softmax")(dec_input)
H = keras.layers.Dropout(dropout_rate)(H)
H = keras.layers.RepeatVector(seq_len)(H)
H = keras.layers.LSTM(embed_size, return_sequences=True)(H)
dec_output = keras.layers.Activation('softmax')(H)
decoder = keras.models.Model(dec_input, dec_output)
decoder.compile(loss="categorical_crossentropy", optimizer="adam")

# test decoder model
prediction = decoder.predict(autoenc_vectors)
print(text_format.tensor_to_string(prediction, model))
