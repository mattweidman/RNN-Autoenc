import random
import re
import os

import gensim

class FileIterator:
    """
    Class that lets you iterate line-by-line through a file,
    returning lines as lists of words. Used by word2vec.
    """

    def __init__(self, filename):
        """
        Choose file path.
        """
        self.filename = filename

    def __iter__(self):
        """
        Each item iterated over is a list of words.
        """
        with open(self.filename, 'r') as f:
            for line in f:
                yield line.split(' ')

class FileBuilder:
    """
    Class for loading and modifying a file.
    """

    def __init__(self, infile, outfile):
        """
        Load a file and write it in outfile. All future
        operations will modify outfile.
        infile: name of file to read from
        outfile: name of file to write to
        """
        with open(infile, 'r') as inf:
            with open(outfile, 'w') as outf:
                for line in inf:
                    outf.write(line)
        self.filename = outfile

    def __getitem__(self, index):
        """
        Find a line at a certain index. To preserve memory,
        this function will iterate line-by-line until it gets
        to the right index, instead of using an array index.
        So indexing a FileBuilder takes linear time.
        """
        with open(self.filename, 'r') as f:
            i = 0
            for i, line in enumerate(f):
                if i == index:
                    return line
            raise IndexError

    def __len__(self):
        """
        Number of lines in file. Like __getitem__(), this
        takes linear time. I don't cache the length because
        there are a lot of functions that change the length
        of the file, and changing length in every function
        that does this would make everything more error prone.
        """
        with open(self.filename, 'r') as f:
            length = 0
            for line in f: length += 1
            return length

    def __copy_to_temp(self):
        """
        Copy the file to a temporary one.
        Returns name of file copied.
        """
        randnum = str(random.getrandbits(128))
        tempname = "cache/temp" + randnum
        with open(self.filename, 'r') as f:
            with open(tempname, 'w') as temp:
                for line in f:
                    temp.write(line)
        return tempname

    def remove_lines(self, start, end):
        """
        Remove lines from start to end.
        start: index of first line to remove (inclusive)
        end: index after last line to remove (exclusive)
        """
        # write to temporary file
        tempname = self.__copy_to_temp()
        # move from temporary file to original file
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for i, line in enumerate(temp):
                    if i < start or i >= end:
                        f.write(line)
        # delete temporary file
        os.remove(tempname)

    def remove_lines_within(self, startStr, endStr):
        """
        Begins removing lines when startStr is seen,
        and stops removing lines when endStr is seen.
        Does so throught the entire document.
        startStr: indicator to start removing strings
        endStr: indicator to stop removing strings
        """
        tempname = self.__copy_to_temp()
        isRemoving = False
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
                    if startStr in line:
                        isRemoving = True
                    if not isRemoving:
                        f.write(line)
                    if endStr in line:
                        isRemoving = False
        os.remove(tempname)

    def remove_number_lines(self, minLineLen=None):
        """
        If a line contains only numeric characters, remove it
        minLineLen: if given, will only remove lines shorter than this
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
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
                        f.write(line)
        os.remove(tempname)

    def remove_empty_lines(self):
        """
        Remove lines that are just whitespace
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
                    if line.strip() != "":
                        f.write(line)
        os.remove(tempname)

    def remove_repeated_empty_lines(self):
        """
        Make sure there is no repeated sequence of empty lines longer
        than one.
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                last_line_was_empty = False
                for line in temp:
                    this_line_is_empty = line.strip() == ""
                    if not (last_line_was_empty and this_line_is_empty):
                        f.write(line)
                    last_line_was_empty = this_line_is_empty
        os.remove(tempname)

    def sub(self, regex, replacement):
        """
        Go through each line, and if regex is found, replace it
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
                    f.write(re.sub(regex, replacement, line))
        os.remove(tempname)

    def to_words(self, char_words):
        """
        Convert any chosen character into a word so that
        it has spaces on either side
        char_words: list of characters to convert to words
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
                    j = 0
                    while j < len(line):
                        if line[j] in char_words:
                            if j>0 and line[j-1] != ' ':
                                line = line[:j] + ' ' + line[j:]
                                j += 1
                            if j < len(line)-1 and \
                                    line[j+1] not in [' ','\n','\r']:
                                line = line[:j+1] + ' ' + line[j+1:]
                                j += 1
                        j += 1
                    f.write(line)
        os.remove(tempname)

    def to_words_apostrophes(self):
        """
        Convert apostrophes to words, but only the ones used
        as quotes - not apostrophes used in contractions.
        """
        tempname = self.__copy_to_temp()
        with open(self.filename, 'w') as f:
            with open(tempname, 'r') as temp:
                for line in temp:
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
                    f.write(line)
        os.remove(tempname)

    '''def write(self, fileName):
        """
        Write lines to file
        """
        with open(fileName, 'w') as f:
            for line in self.lines:
                f.write(line)'''

    def get_model(self, min_count=5, size=100, workers=3, fileName=None):
        """
        Constructs word2vec model
        min_count: forgets about words less frequent than this
        size: output vector size
        workers: number of worker threads running at once
        fileName: file to write model to. If none, doesn't write.
        """
        iterator = FileIterator(self.filename)
        model = gensim.models.Word2Vec(iterator,
            min_count=min_count, size=size, workers=workers)
        if fileName is not None:
            model.save(fileName)
        return model

class TextConverter:
    """
    Converts text into other representations, including numpy arrays.
    Numpy arrays are very memory-intensive, so we need functions
    from this array to take a few pieces of the text at a time.
    """

    def __init__(self, filename):
        """
        filename: name of file containing text
        """
        self.filename = filename

    def get_lines(self, line_nums):
        """
        Gets the list of strings that are at the specified line numbers.
        line_nums: list of line numbers
        """
        # create list of (line number, line string) tuples
        num_strs = []
        line_num_set = set(line_nums) # slight optimization
        with open(self.filename, 'r') as f:
            for i, line in enumerate(f):
                if i in line_num_set:
                    num_strs.append((i,line))

        # way of finding position requested by user, given a line number
        line_num_to_index = dict((x,i) for i,x in enumerate(line_nums))

        # put all strings into a list
        line_strs = [""] * len(line_nums)
        for line_num, line_str in num_strs:
            index = line_num_to_index[line_num]
            line_strs[index] = line_str
        return line_strs
