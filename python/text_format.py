import re

class FileBuilder:
    """
    Class for loading and modifying a file.
    """

    def __init__(self, fileName):
        """
        Load a file.
        filename: name of file to load
        """
        with open(fileName, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """
        Find a line at a certain index
        """
        return self.lines[index]

    def __len__(self):
        """
        Number of lines in file
        """
        return len(self.lines)

    def remove_lines(self, start, end):
        """
        Remove lines from start to end
        start: index of first line to remove (inclusive)
        end: index after last line to remove (exclusive)
        """
        self.lines[start:end] = []

    def remove_lines_within(self, startStr, endStr):
        """
        Begins removing lines when startStr is seen,
        and stops removing lines when endStr is seen.
        Does so throught the entire document.
        startStr: indicator to start removing strings
        endStr: indicator to stop removing strings
        """
        newLines = []
        isRemoving = False
        for line in self.lines:
            if startStr in line:
                isRemoving = True
            if not isRemoving:
                newLines.append(line)
            if endStr in line:
                isRemoving = False
        self.lines = newLines

    def remove_number_lines(self, minLineLen=None):
        """
        If a line contains only numeric characters, remove it
        minLineLen: if given, will only remove lines shorter than this
        """
        newLines = []
        for line in self.lines:
            all_nums = True
            found_num = False
            for c in line.strip():
                if c < '0' or c > '9':
                    all_nums = False
                    break
                else:
                    found_num = True
            all_nums = all_nums and found_num
            if not all_nums or (minLineLen != None and \
                    len(line.strip()) >= minLineLen):
                newLines.append(line)
        self.lines = newLines

    def remove_empty_lines(self):
        """
        Remove lines that are just whitespace
        """
        newLines = []
        for line in self.lines:
            if line.strip() != "":
                newLines.append(line)
        self.lines = newLines

    def remove_repeated_empty_lines(self):
        """
        Make sure there is no repeated sequence of empty lines longer
        than one.
        """
        newLines = []
        last_line_was_empty = False
        for line in self.lines:
            this_line_is_empty = line.strip() == ""
            if not (last_line_was_empty and this_line_is_empty):
                newLines.append(line)
            last_line_was_empty = this_line_is_empty
        self.lines = newLines

    def sub(self, regex, replacement):
        """
        Go through each line, and if regex is found, replace it
        """
        for i, line in enumerate(self.lines):
            self.lines[i] = re.sub(regex, replacement, line)

    def to_words(self, char_words):
        """
        Convert any chosen character into a word so that
        it has spaces on either side
        char_words: list of characters to convert to words
        """
        for i, line in enumerate(self.lines):
            j = 0
            while j < len(line):
                if line[j] in char_words:
                    if j>0 and line[j-1] != ' ':
                        line = line[:j] + ' ' + line[j:]
                        j += 1
                    if j < len(line)-1 and line[j+1] not in [' ','\n','\r']:
                        line = line[:j+1] + ' ' + line[j+1:]
                        j += 1
                j += 1
            self.lines[i] = line

    def to_words_apostrophes(self):
        """
        Convert apostrophes to words, but only the ones used
        as quotes - not apostrophes used in contractions.
        """
        for i, line in enumerate(self.lines):
            j = 0
            while j < len(line):
                if line[j] == "'":
                    if j > 0 and j < len(line)-1 and \
                            line[j-1] != ' ' and \
                            line[j+1] not in [' ','\n','\r']:
                        pass
                    elif j > 0 and line[j-1] != ' ':
                        line = line[:j] + ' ' + line[j:]
                        j += 1
                    elif j < len(line)-1 and \
                            line[j+1] not in [' ','\n','\r']:
                        line = line[:j+1] + ' ' + line[j+1:]
                        j += 1
                j += 1
            self.lines[i] = line

    def write(self, fileName):
        """
        Write lines to file
        """
        with open(fileName, 'w') as f:
            for line in self.lines:
                f.write(line)
