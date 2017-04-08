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
