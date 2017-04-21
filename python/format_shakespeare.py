from text_format import FileIterator
from text_format import FileBuilder
from text_format import DataSet
from text_format import matrix_to_string, tensor_to_string

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

if __name__ == "__main__":
    #modifyShakespeare(inname, outname)
    shakespeare2vec(outname)
