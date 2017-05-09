from text_format import FileBuilder
from text_format import DataSet
from text_format import string_to_matrix

import numpy as np

outfile = "cache/test"

def test_load():
    builder = FileBuilder("../test_text/test_text_1.txt", outfile)
    assert builder[0] == 'abcdefg\n'
    assert builder[4] == '3456789\n'
    assert len(builder) == 5

def test_remove_lines():
    builder = FileBuilder("../test_text/test_text_1.txt", outfile)
    builder.remove_lines(3,5)
    assert len(builder) == 3
    builder.remove_lines(0,2)
    assert builder[0] == 'opqrstu\n'
    assert len(builder) == 1

def test_remove_within():
    builder = FileBuilder("../test_text/remove.txt", outfile)
    builder.remove_lines_within('<<', '>>')
    assert builder[0] == 'abcdefg\n'
    assert builder[1] == 'hijklmn\n'
    assert builder[2] == 'opqrstu\n'
    assert len(builder) == 3

def test_remove_number_lines():
    builder = FileBuilder("../test_text/number_lines.txt", outfile)
    builder.remove_number_lines(4)
    assert builder[0] == 'abc\n'
    assert builder[1] == '\n'
    assert builder[2] == '   \n'
    assert builder[3] == 'six\n'
    assert builder[4] == '12345\n'
    assert builder[5] == '1234\n'
    assert builder[6] == 'q\n'
    builder = FileBuilder("../test_text/number_lines.txt", outfile)
    builder.remove_number_lines()
    assert builder[0] == 'abc\n'
    assert builder[1] == '\n'
    assert builder[2] == '   \n'
    assert builder[3] == 'six\n'
    assert builder[4] == 'q\n'

def test_sub():
    builder = FileBuilder("../test_text/sub.txt", outfile)
    builder.sub('[<}`.]', '')
    assert builder[0] == 'abcdefg\n'
    builder = FileBuilder("../test_text/sub.txt", outfile)
    builder.sub('[<}`.]', ' ')
    assert builder[0] == 'abc d ef   g\n'

"""def test_line_breaks():
    builder = FileBuilder("../test_text/line_break.txt", outfile)
    builder.to_words_line_breaks()
    assert builder[0] == 'abc def \n'
    assert builder[1] == 'ghi  \n'
    assert builder[2] == '\n'
    assert builder[3] == '   j \n'
    assert builder[4] == ' \n'
    assert builder[5] == '  \n'
    assert builder[6] == 'k \n'
    assert builder[7] == '\n'
    assert builder[8] == 'l \n' """

def test_to_words():
    builder = FileBuilder("../test_text/to_words.txt", outfile)
    builder.to_words(['!', '@', '#', '$', '%', '^', '&', '(', ')', '\n'])
    assert builder[0] == 'abcdefg \n'
    assert builder[1] == '$ % # & ( ) \n'
    assert builder[2] == '   ^   \n'

def test_apostrophes():
    builder = FileBuilder("../test_text/apostrophes.txt", outfile)
    builder.to_words_apostrophes()
    assert builder[0] == "' hello '\n"
    assert builder[1] == "  ' hello '\n"
    assert builder[2] == "can't\n"
    assert builder[3] == "' can't '\n"
    assert builder[4] == "' ' ' '\n"

def test_remove_empty_lines():
    builder = FileBuilder("../test_text/empty_lines.txt", outfile)
    builder.remove_empty_lines()
    assert builder[0] == 'a\n'
    assert builder[1] == '9\n'

def test_remove_repeated_empty_lines():
    builder = FileBuilder("../test_text/empty_lines_2.txt", outfile)
    builder.remove_repeated_empty_lines()
    assert builder[0] == 'abc\n'
    assert builder[1] == 'def\n'
    assert builder[2] == '\n'
    assert builder[3] == 'ghi\n'
    assert builder[4] == '\n'
    assert builder[5] == 'jkl\n'
    assert builder[6] == '\n'
    assert builder[7] == 'mno\n'

def test_lowercase():
    builder = FileBuilder("../test_text/lowercase.txt", outfile)
    builder.to_lowercase()
    assert builder[0] == 'abcd34*@\n'

def test_strip():
    builder = FileBuilder("../test_text/strip.txt", outfile)
    builder.strip_lines()
    assert builder[0] == "hello\n"
    assert builder[1] == "world\n"
    assert builder[2] == "ok\n"

def test_get_model():
    builder = FileBuilder("../test_text/getmodel.txt", outfile)
    model = builder.get_model(1, 20, 1, "../test_text/test_model.w2v")

def test_get_lines():
    converter = DataSet("../test_text/get_lines.txt", 10, model={})
    assert converter.get_lines([8, 0, 4]) == ["i\n", "a\n", "e\n"]
    assert converter.get_lines(range(2)) == ["a\n", "b\n"]
    assert converter.get_lines([9]) == ["j\n"]
    assert converter.get_lines(range(10)) == ["a\n", "b\n", "c\n", "d\n",
        "e\n", "f\n", "g\n", "h\n", "i\n", "j\n"]
    assert converter.get_lines(range(9,-1,-1)) == ["j\n", "i\n", "h\n",
        "g\n", "f\n", "e\n", "d\n", "c\n", "b\n", "a\n"]
    assert converter.get_lines([]) == []

def test_s2m():
    model = {'a' : np.array([1,0]), 'b' : np.array([0,1]),
        '' : np.zeros((2))}
    s = "a b b a b"
    calc_matrix = string_to_matrix(s, model, 6, 2)
    exp_matrix = np.array([[1,0], [0,1], [0,1], [1,0], [0,1], [0,0]])
    assert np.array_equal(calc_matrix, exp_matrix)

def test_data_partitions():
    dataset = DataSet("../test_text/get_lines.txt", 10, model={},
        part_training=0.5, part_validation=0.3)
    assert len(dataset) == 10
    assert len(dataset.training_indices) == 5
    assert len(dataset.validation_indices) == 3
    assert len(dataset.test_indices) == 2

if __name__ == "__main__":
    test_load()
    test_remove_lines()
    test_remove_within()
    test_remove_number_lines()
    test_sub()
    #test_line_breaks()
    test_to_words()
    test_apostrophes()
    test_remove_empty_lines()
    test_remove_repeated_empty_lines()
    test_lowercase()
    #test_get_model()
    test_get_lines()
    test_s2m()
    test_data_partitions()
    test_strip()
