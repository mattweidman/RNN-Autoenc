import re

inname = "shakes_raw.txt"
outname = "shakes_out.txt"

line_first = 175
line_last = 124368

with open(inname, 'r') as infile:
    with open(outname, 'w') as outfile:
        in_disclaimer = False
        for i, line in enumerate(infile.readlines()):

            # don't write if outside of range
            if i < line_first: continue
            if i >= line_last: continue

            # don't write if inside a disclaimer
            if not in_disclaimer:
                if line[:2] == "<<":
                    in_disclaimer = True
                    continue
            else:
                if ">>" in line:
                    in_disclaimer = False
                continue

            # don't write if line is just a single number
            if len(line) > 0 and len(line) < 6:
                all_nums = True
                for c in line:
                    is_num = c >= '0' and c <= '9'
                    if not is_num:
                        all_nums = False
                        break
                if all_nums:
                    continue

            # remove forbidden characters
            line = re.sub('[<}`]', '', line)
            line = re.sub('[|_]', ' ', line)

            # turn non-alphanumeric characters into words
            char_words = ['!', '"', '&', '(', ')', ',', '-', '.', ':',
                    ';', '?', '[', ']']
            j = 0
            while j < len(line):
                if line[j] in char_words:
                    if j>0 and line[j-1] != ' ':
                        line = line[:j] + ' ' + line[j:]
                        j += 1
                    if j < len(line)-1 and line[j+1] != ' ':
                        line = line[:j+1] + ' ' + line[j+1:]
                        j += 1
                j += 1

            # put spaces for apostrophes used as quotes
            j = 0
            while j < len(line):
                if line[j] == "'":
                    if j > 0 and j < len(line)-1 and \
                            line[j-1] != ' ' and line[j+1] != ' ':
                        pass
                    elif j > 0 and line[j-1] != ' ':
                        line = line[:j] + ' ' + line[j:]
                        j += 1
                    elif j < len(line)-1 and line[j+1] != ' ':
                        line = line[:j+1] + ' ' + line[j+1:]
                        j += 1
                j += 1

            # write line to file
            outfile.write(line)
