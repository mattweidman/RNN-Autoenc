from text_format import FileIterator
from text_format import FileBuilder
from text_format import DataSet
from text_format import matrix_to_string, tensor_to_string

import numpy as np

inname = "../shakespeare/shakes_raw.txt"
outname = "../shakespeare/shakes_out.txt"

line_first = 175
line_last = 124368

def modifyShakespeare(file_in, file_out):
    builder = FileBuilder(file_in, file_out)
    builder.remove_lines(line_last, len(builder))
    builder.remove_lines(0, line_first)
    builder.remove_repeated_empty_lines()
    builder.remove_lines_within("<<", ">>")
    builder.remove_number_lines(6)
    builder.sub('[<}`\r]', '')
    builder.sub('[|_]', ' ')
    char_words = ['!', '"', '&', '(', ')', ',', '-', '.', ':',
            ';', '?', '[', ']', '\n']
    builder.to_words(char_words)
    builder.to_words_apostrophes()
    builder.to_lowercase()

def shakespeare2vec(filename):
    # set up problem
    line_nums = [9, 7, 5, 28, 20]
    line_len = 16
    embed_size = 100

    # train word2vec model
    iterator = FileIterator(filename)
    #model = iterator.get_model(size=embed_size, workers=4)

    # convert to tensor
    converter = DataSet(filename, embed_size,
        part_training=0.5, part_validation=0.3)
    tensor = converter.get_tensor(line_nums)

    # convert back to sentence
    model = converter.model
    print(tensor_to_string(tensor, model))

    # get random data
    tensor = converter.get_training_data(5)
    print(tensor_to_string(tensor, model))
    tensor = converter.get_validation_data(3)
    print(tensor_to_string(tensor, model))
    tensor = converter.get_test_data(2)
    print(tensor_to_string(tensor, model))

def shakespeare_long_vec(filename):
    # construct dataset
    embed_size = 100
    dataset = DataSet(filename, embed_size)

    # test word_to_long_vector
    word = dataset.word_list[36]
    long_vec = dataset.word_to_long_vector(word)
    expected = np.zeros((len(dataset.word_list)))
    expected[36] = 1
    assert np.allclose(long_vec, expected)

    # test line_to_long_matrix
    line_words = dataset.word_list[35:40] + ["unexpected_word"]
    line = ' '.join(line_words)
    long_matrix = dataset.line_to_long_matrix(line)
    for i, row in enumerate(long_matrix[:5]):
        expected = np.zeros((len(dataset.word_list)))
        expected[35+i] = 1
        assert np.allclose(row, expected)
    for i in range(5, long_matrix.shape[0]):
        assert np.allclose(long_matrix[i],
            dataset.word_to_long_vector(dataset.padding_word))

    # test line_nums_to_long_tensor
    line_nums = [9, 7, 5, 28, 20]
    tensor = dataset.line_nums_to_long_tensor(line_nums)
    outp_string = dataset.long_tensor_to_string(tensor)
    print(outp_string)

if __name__ == "__main__":
    #modifyShakespeare(inname, outname)
    #shakespeare2vec(outname)
    shakespeare_long_vec(outname)
