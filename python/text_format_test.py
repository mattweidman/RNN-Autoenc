from text_format import FileBuilder

def test_load():
    builder = FileBuilder("../test_text/test_text_1.txt")
    assert builder[0] == 'abcdefg\n'
    assert builder[4] == '3456789\n'
    assert len(builder) == 5

def test_remove_lines():
    builder = FileBuilder("../test_text/test_text_1.txt")
    builder.remove_lines(3,5)
    assert len(builder) == 3
    builder.remove_lines(0,2)
    assert builder[0] == 'opqrstu\n'
    assert len(builder) == 1

def test_remove_within():
    builder = FileBuilder("../test_text/remove.txt")
    builder.remove_lines_within('<<', '>>')
    assert builder[0] == 'abcdefg\n'
    assert builder[1] == 'hijklmn\n'
    assert builder[2] == 'opqrstu\n'
    assert len(builder) == 3

if __name__ == "__main__":
    test_load()
    test_remove_lines()
    test_remove_within()
