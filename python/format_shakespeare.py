from text_format import FileIterator
from text_format import FileBuilder
from text_format import TextConverter

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
    builder.sub('[<}`]', '')
    builder.sub('[|_]', ' ')
    char_words = ['!', '"', '&', '(', ')', ',', '-', '.', ':',
            ';', '?', '[', ']']
    builder.to_words(char_words)
    builder.to_words_apostrophes()

def shakespeare2vec(filename):
    # set up problem
    line_nums = [9, 7, 5]
    line_len = 12
    embed_size = 100

    # train word2vec model
    iterator = FileIterator(filename)
    model = iterator.get_model(size=embed_size, workers=4)

    # convert to tensor
    converter = TextConverter(filename)
    tensor = converter.get_tensor(line_nums, model, line_len, embed_size)

    # convert back to sentence
    output_text = ""
    for i in range(len(line_nums)):
        sentence_arr = []
        for vec in tensor[i]:
           word = model.similar_by_vector(vec, topn=1)
           sentence_arr.append(word)
        sentence = ' '.join([tup[0][0] for tup in sentence_arr])
        output_text += sentence
    print(output_text)

if __name__ == "__main__":
    #modifyShakespeare(inname, outname)
    shakespeare2vec(outname)
