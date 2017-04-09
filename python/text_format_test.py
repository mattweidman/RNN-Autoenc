from text_format import FileBuilder

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

def test_to_words():
    builder = FileBuilder("../test_text/to_words.txt", outfile)
    builder.to_words(['!', '@', '#', '$', '%', '^', '&', '(', ')'])
    assert builder[0] == 'abcdefg\n'
    assert builder[1] == '$ % # & ( )\n'
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

def test_get_model():
    builder = FileBuilder("../test_text/getmodel.txt", outfile)
    model = builder.get_model(1, 20, 1, "../test_text/test_model.w2v")

if __name__ == "__main__":
    test_load()
    test_remove_lines()
    test_remove_within()
    test_remove_number_lines()
    test_sub()
    test_to_words()
    test_apostrophes()
    test_remove_empty_lines()
    test_remove_repeated_empty_lines()
    test_get_model()
