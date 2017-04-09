from text_format import FileBuilder

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

if __name__ == "__main__":
    modifyShakespeare(inname, outname)
