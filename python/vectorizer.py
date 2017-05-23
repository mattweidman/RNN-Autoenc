# python file for converting words into vectors and back
import numpy as np

# sorted list of words in text
def get_vocab(filename):
    with open(filename, 'r') as rfile:
        vocab = set()
        for line in rfile.readlines():
            vocab.update(line.split())
        return sorted(set(vocab))

# sorted list of characters in text
def get_alphabet(filename):
    with open(filename, 'r') as rfile:
        alphabet = set()
        for line in rfile.readlines():
            alphabet.update(list(line))
        return alphabet

# word -> vector mapping
class WordToOnehot:
    def __init__(self, vocab):
        self.word_to_index = dict((s,i) for i,s in enumerate(vocab))
    def __getitem__(self, word):
        if word not in self.word_to_index:
            raise KeyError(str(word) + " has no one-hot representation")
        index = self.word_to_index[word]
        vec = np.zeros((len(self.word_to_index)))
        vec[index] = 1
        return vec

# vector -> word mapping
class OnehotToWord:
    def __init__(self, vocab):
        self.vocab = vocab
    def __getitem__(self, vector):
        if vector.shape != (len(vocab),):
            raise KeyError("Vector should be size of vocabulary. Expected "\
                + str((len(vocab),)) + " but got " + str(vector.shape))
        index = np.argmax(vector)
        return self.vocab[index]

# convert string of words into a one-hot word matrix
# s: string to convert
# word_to_onehot: object of class WordToOnehot
def string_to_onehot_word_matrix(s, word_to_onehot):
    return np.array([word_to_onehot[word] for word in s.split()])

# convert a one-hot word matrix to a string
# matx: matrix to convert
# onehot_to_word: object of class OnehotToWord
def onehot_word_matrix_to_string(matx, onehot_to_word):
    return ' '.join([onehot_to_word[vec] for vec in matx])

# convert list of strings to one-hot word tensor
# s: list of strings
# word_to_onehot: object of class WordToOnehot
def strs_to_onehot_word_tensor(strs, word_to_onehot):
    return np.array([string_to_onehot_word_matrix(s, word_to_onehot) \
        for s in strs])

if __name__ == "__main__":
    filename = "../shakespeare/shakes_out.txt"
    vocab = get_vocab(filename)
    word_to_onehot = WordToOnehot(vocab)
    onehot_to_word = OnehotToWord(vocab)
    phrase = "! , zwagger'd"
    matx = string_to_onehot_word_matrix("! , zwagger'd", word_to_onehot)
    assert matx.shape == (3, len(vocab))
    predict_phrase = onehot_word_matrix_to_string(matx, onehot_to_word)
    assert predict_phrase == phrase

    phrases = ["zone zounds zodiacs", "you're, you'll, you'st, you'ld",
        "xanthippe"]
